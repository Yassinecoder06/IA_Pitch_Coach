import { POSE, type Landmark, type PoseFrame } from './types';

export type PostureIssue = 'slouching' | 'shoulders_tilted' | 'arms_crossed' | 'hand_on_face' | 'hands_hidden';

export interface PostureReading {
  /** 0..100, 100 = open, upright, level. */
  score: number;
  issues: PostureIssue[];
  shoulderTiltDeg: number;
  /** Head height above the shoulders relative to shoulder width (drops when slouching). */
  neckRatio: number;
  bodyPresent: boolean;
}

export interface PostureBaseline {
  neckRatio: number;
}

const visible = (l?: Landmark, min = 0.5) => !!l && (l.visibility ?? 1) >= min;
const dist = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y);

const PENALTY: Record<PostureIssue, number> = {
  slouching: 30,
  arms_crossed: 25,
  hand_on_face: 20,
  shoulders_tilted: 15,
  hands_hidden: 10,
};

export function analysePosture(pose: PoseFrame | null, baseline?: PostureBaseline): PostureReading {
  const lm = pose?.landmarks;
  const ls = lm?.[POSE.leftShoulder];
  const rs = lm?.[POSE.rightShoulder];
  if (!lm || !visible(ls) || !visible(rs)) {
    return { score: 0, issues: [], shoulderTiltDeg: 0, neckRatio: 0, bodyPresent: false };
  }
  const nose = lm[POSE.nose];
  const shoulderWidth = Math.max(dist(ls!, rs!), 1e-3);
  const mid = { x: (ls!.x + rs!.x) / 2, y: (ls!.y + rs!.y) / 2 };
  const issues: PostureIssue[] = [];

  // Shoulder tilt, folded into 0..90 so it does not depend on left/right order.
  let tilt = Math.abs((Math.atan2(rs!.y - ls!.y, rs!.x - ls!.x) * 180) / Math.PI);
  if (tilt > 90) tilt = 180 - tilt;
  if (tilt > 8) issues.push('shoulders_tilted');

  const neckRatio = visible(nose, 0.3) ? (mid.y - nose.y) / shoulderWidth : 0;
  const expectedNeck = baseline?.neckRatio ?? 0.55;
  if (neckRatio > 0 && neckRatio < expectedNeck * 0.75) issues.push('slouching');

  const lw = lm[POSE.leftWrist];
  const rw = lm[POSE.rightWrist];
  const lwVis = visible(lw, 0.3);
  const rwVis = visible(rw, 0.3);
  const lh = lm[POSE.leftHip];
  const rh = lm[POSE.rightHip];
  const hipY = visible(lh, 0.3) && visible(rh, 0.3) ? (lh.y + rh.y) / 2 : mid.y + shoulderWidth * 1.6;

  if (lwVis && rwVis) {
    const side = (p: Landmark) => Math.sign(p.x - mid.x);
    const inTorso = (p: Landmark) => p.y > mid.y - shoulderWidth * 0.1 && p.y < hipY;
    // Crossed: each wrist sits on the opposite side of the body midline from its own shoulder...
    const pastMidline = side(lw) === -side(ls!) && side(rw) === -side(rs!);
    // ...or, as the model usually reports it (the hands are hidden under the opposite arm), both
    // forearms run inward and level from elbows at the sides, ending near the midline. Clasped
    // hands differ: the wrists hang below the elbows.
    const folded = (wrist: Landmark, elbow: Landmark | undefined, shoulder: Landmark) =>
      visible(elbow, 0.3) &&
      side(shoulder) * (elbow!.x - wrist.x) > shoulderWidth * 0.4 &&
      Math.abs(wrist.x - mid.x) < shoulderWidth * 0.6 &&
      wrist.y < elbow!.y + shoulderWidth * 0.25;
    const bothFolded = folded(lw, lm[POSE.leftElbow], ls!) && folded(rw, lm[POSE.rightElbow], rs!);
    if ((pastMidline || bothFolded) && inTorso(lw) && inTorso(rw)) issues.push('arms_crossed');
  }
  if (visible(nose, 0.3)) {
    const nearFace = (p: Landmark, vis: boolean) => vis && dist(p, nose) < shoulderWidth * 0.45;
    if (nearFace(lw, lwVis) || nearFace(rw, rwVis)) issues.push('hand_on_face');
  }
  if (!lwVis && !rwVis) issues.push('hands_hidden');

  const score = Math.max(0, 100 - issues.reduce((s, i) => s + PENALTY[i], 0));
  return { score, issues, shoulderTiltDeg: tilt, neckRatio, bodyPresent: true };
}

/**
 * Majority vote over the last few pose frames, so one noisy landmark (often an occluded wrist)
 * does not make issues flicker on and off.
 */
export class PostureSmoother {
  private recent: PostureIssue[][] = [];

  constructor(private readonly frames = 9) {}

  update(reading: PostureReading): PostureReading {
    if (!reading.bodyPresent) {
      this.recent = [];
      return reading;
    }
    this.recent.push(reading.issues);
    if (this.recent.length > this.frames) this.recent.shift();
    const counts = new Map<PostureIssue, number>();
    for (const frame of this.recent) for (const i of frame) counts.set(i, (counts.get(i) ?? 0) + 1);
    const issues = [...counts].filter(([, n]) => n * 2 > this.recent.length).map(([i]) => i);
    const score = Math.max(0, 100 - issues.reduce((s, i) => s + PENALTY[i], 0));
    return { ...reading, issues, score };
  }
}
