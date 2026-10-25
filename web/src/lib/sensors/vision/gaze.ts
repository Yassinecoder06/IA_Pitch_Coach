import { FACE_POINTS as P, type FaceFrame, type Landmark } from './types';

export type GazeState = 'good' | 'weak' | 'lost';

export interface GazeReading {
  /** Smoothed eye-contact score 0..1. */
  score: number;
  state: GazeState;
  facePresent: boolean;
  yaw: number; // degrees, relative to calibration
  pitch: number;
}

export interface GazeBaseline {
  yaw: number;
  pitch: number;
  irisH: number;
  irisV: number;
}

const DEFAULT_BASELINE: GazeBaseline = { yaw: 0, pitch: 0, irisH: 0.5, irisV: 0.5 };
const GOOD = 0.6;
const WEAK = 0.3;
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/** Yaw / pitch / roll in degrees from MediaPipe's column-major facial transformation matrix. */
export function headAngles(matrix: number[]): { yaw: number; pitch: number; roll: number } {
  const r = (row: number, col: number) => matrix[col * 4 + row];
  const deg = 180 / Math.PI;
  return {
    pitch: Math.atan2(r(2, 1), r(2, 2)) * deg,
    yaw: Math.atan2(-r(2, 0), Math.hypot(r(2, 1), r(2, 2))) * deg,
    roll: Math.atan2(r(1, 0), r(0, 0)) * deg,
  };
}

function ratio(value: number, a: number, b: number): number {
  const span = b - a;
  return Math.abs(span) < 1e-6 ? 0.5 : (value - a) / span;
}

/** Iris position inside each eye, 0.5 = centred. Averaged over both eyes. */
export function irisPosition(points: Record<number, Landmark>): { h: number; v: number } | null {
  const eyes = [
    [P.rightIris, P.rightEyeOuter, P.rightEyeInner, P.rightEyeTop, P.rightEyeBottom],
    [P.leftIris, P.leftEyeInner, P.leftEyeOuter, P.leftEyeTop, P.leftEyeBottom],
  ];
  const hs: number[] = [];
  const vs: number[] = [];
  for (const [iris, cornerA, cornerB, top, bottom] of eyes) {
    const i = points[iris];
    const a = points[cornerA];
    const b = points[cornerB];
    const t = points[top];
    const btm = points[bottom];
    if (!i || !a || !b || !t || !btm) continue;
    // Corners ordered by image x so the ratio does not depend on mirroring.
    const [left, right] = a.x <= b.x ? [a, b] : [b, a];
    hs.push(ratio(i.x, left.x, right.x));
    vs.push(ratio(i.y, t.y, btm.y));
  }
  if (!hs.length) return null;
  return { h: hs.reduce((s, x) => s + x, 0) / hs.length, v: vs.reduce((s, x) => s + x, 0) / vs.length };
}

/** Instantaneous eye-contact score 0..1 for one face, before smoothing. */
export function rawEyeContact(face: FaceFrame, base: GazeBaseline = DEFAULT_BASELINE): { score: number; yaw: number; pitch: number } {
  const angles = face.matrix ? headAngles(face.matrix) : { yaw: 0, pitch: 0, roll: 0 };
  const yaw = angles.yaw - base.yaw;
  const pitch = angles.pitch - base.pitch;
  const head = clamp01(1 - Math.abs(yaw) / 30) * clamp01(1 - Math.abs(pitch) / 25);

  const iris = irisPosition(face.points);
  const irisScore = iris
    ? clamp01(1 - Math.abs(iris.h - base.irisH) / 0.22) * clamp01(1 - Math.abs(iris.v - base.irisV) / 0.3)
    : head;

  const b = face.blendshapes;
  const avg = (x: string, y: string) => ((b[x] ?? 0) + (b[y] ?? 0)) / 2;
  const away = Math.max(
    avg('eyeLookUpLeft', 'eyeLookUpRight'),
    avg('eyeLookDownLeft', 'eyeLookDownRight'),
    avg('eyeLookOutLeft', 'eyeLookInRight'), // both eyes turned to one side
    avg('eyeLookInLeft', 'eyeLookOutRight'),
  );
  const blendScore = clamp01(1 - away / 0.6);
  const eyesClosed = avg('eyeBlinkLeft', 'eyeBlinkRight') > 0.6;

  // Head pose gates the eye signals: a face turned well away cannot be making eye contact,
  // even when the eye readings are missing or ambiguous.
  const score = eyesClosed ? 0 : (0.55 * irisScore + 0.45 * blendScore) * (0.3 + 0.7 * head);
  return { score, yaw, pitch };
}

export function classify(score: number): GazeState {
  return score >= GOOD ? 'good' : score >= WEAK ? 'weak' : 'lost';
}

/** Smooths per-frame eye contact and keeps the share of time with good eye contact. */
export class GazeTracker {
  private smoothed = 0;
  private baseline: GazeBaseline = DEFAULT_BASELINE;
  private calibration: { yaw: number; pitch: number; h: number; v: number }[] | null = null;
  private goodMs = 0;
  private totalMs = 0;
  private lastT: number | null = null;

  constructor(private readonly alpha = 0.3) {}

  /** Start collecting frames; call finishCalibration() after ~3 s of looking at the lens. */
  startCalibration() {
    this.calibration = [];
  }

  finishCalibration(): GazeBaseline | null {
    const samples = this.calibration;
    this.calibration = null;
    if (!samples || samples.length < 10) return null;
    const mean = (k: 'yaw' | 'pitch' | 'h' | 'v') => samples.reduce((s, x) => s + x[k], 0) / samples.length;
    this.baseline = { yaw: mean('yaw'), pitch: mean('pitch'), irisH: mean('h'), irisV: mean('v') };
    return this.baseline;
  }

  resetStats() {
    this.goodMs = 0;
    this.totalMs = 0;
    this.lastT = null;
  }

  /** Share of tracked time with good eye contact, 0..1. */
  get eyeContactShare(): number {
    return this.totalMs > 0 ? this.goodMs / this.totalMs : 0;
  }

  update(face: FaceFrame | null, tMs: number): GazeReading {
    let raw = 0;
    let yaw = 0;
    let pitch = 0;
    if (face) {
      if (this.calibration && face.matrix) {
        const a = headAngles(face.matrix);
        const iris = irisPosition(face.points);
        if (iris) this.calibration.push({ yaw: a.yaw, pitch: a.pitch, h: iris.h, v: iris.v });
      }
      ({ score: raw, yaw, pitch } = rawEyeContact(face, this.baseline));
    }
    this.smoothed = this.alpha * raw + (1 - this.alpha) * this.smoothed;
    const state = face ? classify(this.smoothed) : 'lost';

    if (this.lastT !== null) {
      const dt = Math.min(500, Math.max(0, tMs - this.lastT)); // ignore long gaps (tab hidden)
      this.totalMs += dt;
      if (state === 'good') this.goodMs += dt;
    }
    this.lastT = tMs;
    return { score: this.smoothed, state, facePresent: !!face, yaw, pitch };
  }
}
