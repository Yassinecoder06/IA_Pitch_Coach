import { describe, expect, it } from 'vitest';
import { GazeTracker, headAngles, irisPosition, rawEyeContact } from '../vision/gaze';
import { analysePosture } from '../vision/posture';
import { MotionTracker } from '../vision/motion';
import { FACE_POINTS as F, POSE, type FaceFrame, type Landmark, type PoseFrame } from '../vision/types';

// ---------- synthetic fixtures ----------
function yawMatrix(deg: number): number[] {
  const a = (deg * Math.PI) / 180;
  // Rotation about Y, column-major 4x4
  return [Math.cos(a), 0, -Math.sin(a), 0, 0, 1, 0, 0, Math.sin(a), 0, Math.cos(a), 0, 0, 0, 0, 1];
}

function face({ irisShift = 0, yaw = 0, blend = {} as Record<string, number> } = {}): FaceFrame {
  const eye = (cx: number) => ({ a: { x: cx - 0.03, y: 0.4 }, b: { x: cx + 0.03, y: 0.4 }, t: { x: cx, y: 0.39 }, btm: { x: cx, y: 0.41 } });
  const r = eye(0.44);
  const l = eye(0.56);
  return {
    points: {
      [F.rightEyeOuter]: r.a, [F.rightEyeInner]: r.b, [F.rightEyeTop]: r.t, [F.rightEyeBottom]: r.btm,
      [F.leftEyeInner]: l.a, [F.leftEyeOuter]: l.b, [F.leftEyeTop]: l.t, [F.leftEyeBottom]: l.btm,
      [F.rightIris]: { x: 0.44 + irisShift, y: 0.4 },
      [F.leftIris]: { x: 0.56 + irisShift, y: 0.4 },
      [F.noseTip]: { x: 0.5, y: 0.45 },
    },
    blendshapes: blend,
    matrix: yawMatrix(yaw),
  };
}

function pose(overrides: Partial<Record<keyof typeof POSE, Landmark>> = {}): PoseFrame {
  const lm: Landmark[] = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0 }));
  const base: Record<keyof typeof POSE, Landmark> = {
    nose: { x: 0.5, y: 0.33, visibility: 1 },
    leftEar: { x: 0.55, y: 0.32, visibility: 1 },
    rightEar: { x: 0.45, y: 0.32, visibility: 1 },
    leftShoulder: { x: 0.65, y: 0.5, visibility: 1 },
    rightShoulder: { x: 0.35, y: 0.5, visibility: 1 },
    leftElbow: { x: 0.68, y: 0.7, visibility: 1 },
    rightElbow: { x: 0.32, y: 0.7, visibility: 1 },
    leftWrist: { x: 0.67, y: 0.85, visibility: 1 },
    rightWrist: { x: 0.33, y: 0.85, visibility: 1 },
    leftHip: { x: 0.6, y: 0.95, visibility: 1 },
    rightHip: { x: 0.4, y: 0.95, visibility: 1 },
  };
  for (const [k, v] of Object.entries({ ...base, ...overrides })) lm[POSE[k as keyof typeof POSE]] = v as Landmark;
  return { landmarks: lm };
}

// ---------- gaze ----------
describe('gaze', () => {
  it('reads yaw from the transformation matrix', () => {
    expect(headAngles(yawMatrix(30)).yaw).toBeCloseTo(30, 3);
    expect(headAngles(yawMatrix(-15)).yaw).toBeCloseTo(-15, 3);
  });

  it('finds a centred iris at 0.5 regardless of corner order', () => {
    expect(irisPosition(face().points)!.h).toBeCloseTo(0.5, 5);
    expect(irisPosition(face({ irisShift: 0.015 }).points)!.h).toBeCloseTo(0.75, 5);
  });

  it('scores looking at the lens high and looking away low', () => {
    expect(rawEyeContact(face()).score).toBeGreaterThan(0.95);
    expect(rawEyeContact(face({ yaw: 40, irisShift: 0.02 })).score).toBeLessThan(0.2);
    expect(rawEyeContact(face({ blend: { eyeLookDownLeft: 0.8, eyeLookDownRight: 0.8 } })).score).toBeLessThan(0.8);
    expect(rawEyeContact(face({ blend: { eyeBlinkLeft: 0.9, eyeBlinkRight: 0.9 } })).score).toBe(0);
  });

  it('smooths, classifies and accumulates good eye-contact time', () => {
    const g = new GazeTracker();
    let r = g.update(face(), 0);
    for (let t = 66; t <= 2000; t += 66) r = g.update(face(), t);
    expect(r.state).toBe('good');
    for (let t = 2066; t <= 4000; t += 66) r = g.update(face({ yaw: 45, irisShift: 0.025 }), t);
    expect(r.state).toBe('lost');
    expect(g.eyeContactShare).toBeGreaterThan(0.4);
    expect(g.eyeContactShare).toBeLessThan(0.6);
    expect(g.update(null, 4066).facePresent).toBe(false);
  });

  it('calibration re-centres a speaker whose camera is off to the side', () => {
    const g = new GazeTracker();
    const offAxis = face({ yaw: 20 }); // laptop camera at an angle: looking at the lens reads as 20°
    expect(rawEyeContact(offAxis).score).toBeLessThan(0.9);
    g.startCalibration();
    for (let t = 0; t < 3000; t += 66) g.update(offAxis, t);
    expect(g.finishCalibration()!.yaw).toBeCloseTo(20, 3);
    let r = g.update(offAxis, 3066);
    for (let t = 3132; t < 5000; t += 66) r = g.update(offAxis, t);
    expect(r.score).toBeGreaterThan(0.9);
  });
});

// ---------- posture ----------
describe('posture', () => {
  it('rates an open upright posture highly', () => {
    const r = analysePosture(pose());
    expect(r.issues).toEqual([]);
    expect(r.score).toBe(100);
  });

  it('detects crossed arms, slouching, a hand on the face and tilted shoulders', () => {
    expect(analysePosture(pose({ leftWrist: { x: 0.42, y: 0.72, visibility: 0.9 }, rightWrist: { x: 0.58, y: 0.72, visibility: 0.9 } })).issues).toContain('arms_crossed');
    expect(analysePosture(pose({ nose: { x: 0.5, y: 0.43, visibility: 1 } })).issues).toContain('slouching');
    expect(analysePosture(pose({ rightWrist: { x: 0.52, y: 0.36, visibility: 1 } })).issues).toContain('hand_on_face');
    expect(analysePosture(pose({ leftShoulder: { x: 0.65, y: 0.57, visibility: 1 } })).issues).toContain('shoulders_tilted');
  });

  it('is independent of mirroring', () => {
    const mirrored = pose();
    mirrored.landmarks = mirrored.landmarks.map(l => ({ ...l, x: 1 - l.x }));
    expect(analysePosture(mirrored).issues).toEqual([]);
  });

  it('reports no body when shoulders are not visible', () => {
    expect(analysePosture(null).bodyPresent).toBe(false);
    expect(analysePosture(pose({ leftShoulder: { x: 0.6, y: 0.5, visibility: 0.1 } })).bodyPresent).toBe(false);
  });
});

// ---------- motion ----------
describe('motion', () => {
  const run = (frame: (t: number, i: number) => PoseFrame, seconds = 3) => {
    const m = new MotionTracker();
    let r = m.update(frame(0, 0), 0);
    for (let i = 1; i < seconds * 15; i++) r = m.update(frame(i * 66, i), i * 66);
    return r;
  };

  it('a still speaker barely registers', () => {
    const r = run(() => pose());
    expect(r.fidget).toBeLessThan(0.05);
    expect(r.sway).toBeLessThan(0.05);
  });

  it('detects fidgeting hands', () => {
    const r = run((_t, i) => {
      const d = i % 2 ? 0.02 : -0.02;
      return pose({ leftWrist: { x: 0.67 + d, y: 0.85, visibility: 1 }, rightWrist: { x: 0.33 - d, y: 0.85, visibility: 1 } });
    });
    expect(r.fidget).toBeGreaterThan(0.5);
  });

  it('detects swaying', () => {
    const r = run(t => {
      const dx = 0.06 * Math.sin((2 * Math.PI * t) / 1500);
      return pose({ leftShoulder: { x: 0.65 + dx, y: 0.5, visibility: 1 }, rightShoulder: { x: 0.35 + dx, y: 0.5, visibility: 1 } });
    }, 4);
    expect(r.sway).toBeGreaterThan(0.5);
  });

  it('measures smooth gestures in the gesture zone as energy, not fidgeting', () => {
    const r = run(t => {
      const y = 0.65 + 0.1 * Math.sin((2 * Math.PI * t) / 1200);
      return pose({ leftWrist: { x: 0.6, y, visibility: 1 }, rightWrist: { x: 0.4, y, visibility: 1 } });
    });
    expect(r.gestureEnergy).toBeGreaterThan(0.2);
    expect(r.fidget).toBeLessThan(0.5);
  });
});
