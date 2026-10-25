import { POSE, type Landmark, type PoseFrame } from './types';

export interface MotionReading {
  /** 0..1: fast, small, repetitive hand/shoulder movement (fidgeting). */
  fidget: number;
  /** 0..1: side-to-side body sway over the last few seconds. */
  sway: number;
  /** 0..1: purposeful hand movement inside the gesture zone (chest to waist). */
  gestureEnergy: number;
}

interface Sample {
  t: number;
  mid: { x: number; y: number };
  wrists: (Landmark | null)[];
  width: number;
}

const WINDOW_MS = 4000;
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const visible = (l?: Landmark) => !!l && (l.visibility ?? 1) >= 0.4;

function std(values: number[]): number {
  if (values.length < 2) return 0;
  const m = values.reduce((s, x) => s + x, 0) / values.length;
  return Math.sqrt(values.reduce((s, x) => s + (x - m) ** 2, 0) / values.length);
}

/**
 * Body-motion features from a stream of pose frames. Distances are normalised by shoulder
 * width so they do not depend on how far the speaker sits from the camera.
 */
export class MotionTracker {
  private samples: Sample[] = [];

  update(pose: PoseFrame | null, tMs: number): MotionReading {
    const lm = pose?.landmarks;
    const ls = lm?.[POSE.leftShoulder];
    const rs = lm?.[POSE.rightShoulder];
    if (lm && visible(ls) && visible(rs)) {
      this.samples.push({
        t: tMs,
        mid: { x: (ls!.x + rs!.x) / 2, y: (ls!.y + rs!.y) / 2 },
        width: Math.max(Math.hypot(ls!.x - rs!.x, ls!.y - rs!.y), 1e-3),
        wrists: [POSE.leftWrist, POSE.rightWrist].map(i => (visible(lm[i]) ? lm[i] : null)),
      });
    }
    while (this.samples.length && this.samples[0].t < tMs - WINDOW_MS) this.samples.shift();
    return this.reading();
  }

  private reading(): MotionReading {
    const s = this.samples;
    if (s.length < 5) return { fidget: 0, sway: 0, gestureEnergy: 0 };
    const width = s.reduce((a, x) => a + x.width, 0) / s.length;

    // Sway: spread of the shoulder midpoint, relative to shoulder width.
    const sway = clamp01(std(s.map(x => x.mid.x)) / width / 0.15);

    // Wrist movement split into jitter (3-point curvature: high only for frame-to-frame
    // back-and-forth) and gesture speed.
    let jitter = 0;
    let gesture = 0;
    let n = 0;
    for (let i = 2; i < s.length - 2; i++) {
      for (let w = 0; w < 2; w++) {
        const pts = [s[i - 2], s[i - 1], s[i], s[i + 1], s[i + 2]].map(x => x.wrists[w]);
        if (pts.some(p => !p)) continue;
        const dt = Math.max(1, s[i + 1].t - s[i - 1].t) / 1000;
        const speed = Math.hypot(pts[3]!.x - pts[1]!.x, pts[3]!.y - pts[1]!.y) / width / dt;
        const wobble =
          Math.hypot(pts[2]!.x - (pts[1]!.x + pts[3]!.x) / 2, pts[2]!.y - (pts[1]!.y + pts[3]!.y) / 2) / width;
        const p = pts[2]!;
        const inGestureZone = p.y > s[i].mid.y && p.y < s[i].mid.y + width * 1.6 && Math.abs(p.x - s[i].mid.x) < width * 1.5;
        jitter += wobble;
        if (inGestureZone) gesture += speed;
        n++;
      }
    }
    if (!n) return { fidget: 0, sway, gestureEnergy: 0 };
    return {
      fidget: clamp01(jitter / n / 0.08),
      sway,
      gestureEnergy: clamp01(gesture / n / 1.5),
    };
  }
}
