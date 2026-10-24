/// <reference lib="webworker" />
/**
 * Runs MediaPipe Face + Pose landmarkers off the main thread.
 * Receives ImageBitmaps, returns compact landmark data (no pixels ever leave this worker
 * except as numbers, and nothing leaves the browser).
 */

// MediaPipe loads its WASM glue with importScripts(). In module workers that function
// exists but throws, so it is always replaced by a synchronous fetch + global eval.
const scope = self as unknown as { importScripts?: (...urls: string[]) => void };
{
  scope.importScripts = (...urls: string[]) => {
    for (const url of urls) {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', url, false);
      xhr.send();
      if (xhr.status >= 400) throw new Error(`importScripts failed for ${url}: ${xhr.status}`);
      (0, eval)(xhr.responseText);
    }
  };
}

import { FaceLandmarker, FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import { FACE_POINT_INDICES, GAZE_BLENDSHAPES, type FaceFrame, type PoseFrame } from './types';

export type VisionRequest =
  | { type: 'init'; base: string; face: boolean; pose: boolean }
  | { type: 'frame'; bitmap: ImageBitmap; t: number };

export type VisionResponse =
  | { type: 'ready'; delegate: 'GPU' | 'CPU' }
  | { type: 'error'; message: string }
  | { type: 'result'; t: number; face: FaceFrame | null; pose: PoseFrame | null; inferenceMs: number };

let face: FaceLandmarker | null = null;
let pose: PoseLandmarker | null = null;
let lastT = -1;
let config: { base: string; face: boolean; pose: boolean } | null = null;
let delegate: 'GPU' | 'CPU' = 'GPU';

// Delegate auto-tuning: a GPU delegate on software GL (headless, blocklisted drivers) can be far
// slower than WASM on the CPU, so the first frames are timed and the faster delegate is kept.
const PROBE_FRAMES = 15;
const SLOW_GPU_MS = 80;
let probe: { phase: 'gpu' | 'cpu' | 'done'; times: number[]; gpuMedian: number } = { phase: 'gpu', times: [], gpuMedian: 0 };
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

async function switchDelegate(next: 'GPU' | 'CPU') {
  if (!config) return;
  const [f, p] = await create(config.base, config.face, config.pose, next);
  face?.close();
  pose?.close();
  face = f;
  pose = p;
  delegate = next;
  post({ type: 'ready', delegate });
}

async function tune(ms: number) {
  if (probe.phase === 'done') return;
  probe.times.push(ms);
  if (probe.times.length < PROBE_FRAMES) return;
  const m = median(probe.times.slice(3)); // skip warm-up frames
  if (probe.phase === 'gpu') {
    if (delegate !== 'GPU' || m <= SLOW_GPU_MS) {
      probe.phase = 'done';
      return;
    }
    probe = { phase: 'cpu', times: [], gpuMedian: m };
    await switchDelegate('CPU');
  } else {
    probe.phase = 'done';
    if (m > probe.gpuMedian) await switchDelegate('GPU');
  }
}

async function create(base: string, wantFace: boolean, wantPose: boolean, delegate: 'GPU' | 'CPU') {
  const fileset = await FilesetResolver.forVisionTasks(`${base}/wasm`);
  const [f, p] = await Promise.all([
    wantFace
      ? FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: `${base}/models/face_landmarker.task`, delegate },
          runningMode: 'VIDEO',
          numFaces: 1,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        })
      : Promise.resolve(null),
    wantPose
      ? PoseLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: `${base}/models/pose_landmarker_lite.task`, delegate },
          runningMode: 'VIDEO',
          numPoses: 1,
        })
      : Promise.resolve(null),
  ]);
  return [f, p] as const;
}

function compactFace(result: ReturnType<FaceLandmarker['detectForVideo']>): FaceFrame | null {
  const lm = result.faceLandmarks?.[0];
  if (!lm) return null;
  const points: FaceFrame['points'] = {};
  for (const i of FACE_POINT_INDICES) if (lm[i]) points[i] = { x: lm[i].x, y: lm[i].y, z: lm[i].z };
  const blendshapes: Record<string, number> = {};
  for (const c of result.faceBlendshapes?.[0]?.categories ?? []) {
    if (GAZE_BLENDSHAPES.includes(c.categoryName)) blendshapes[c.categoryName] = c.score;
  }
  const matrix = result.facialTransformationMatrixes?.[0]?.data;
  return { points, blendshapes, matrix: matrix ? Array.from(matrix) : undefined };
}

function compactPose(result: ReturnType<PoseLandmarker['detectForVideo']>): PoseFrame | null {
  const lm = result.landmarks?.[0];
  if (!lm) return null;
  return { landmarks: lm.map(l => ({ x: l.x, y: l.y, z: l.z, visibility: l.visibility })) };
}

const post = (msg: VisionResponse) => (self as unknown as Worker).postMessage(msg);

self.onmessage = async (event: MessageEvent<VisionRequest>) => {
  const msg = event.data;
  if (msg.type === 'init') {
    try {
      config = { base: msg.base, face: msg.face, pose: msg.pose };
      delegate = 'GPU';
      try {
        [face, pose] = await create(msg.base, msg.face, msg.pose, 'GPU');
      } catch {
        delegate = 'CPU'; // no WebGL2 in this worker (e.g. headless or blocklisted GPU)
        [face, pose] = await create(msg.base, msg.face, msg.pose, 'CPU');
      }
      post({ type: 'ready', delegate });
    } catch (e) {
      post({ type: 'error', message: e instanceof Error ? e.message : String(e) });
    }
    return;
  }

  if (msg.type === 'frame') {
    const { bitmap } = msg;
    // MediaPipe requires strictly increasing timestamps per landmarker.
    const t = Math.max(msg.t, lastT + 1);
    lastT = t;
    const start = performance.now();
    try {
      const f = face ? compactFace(face.detectForVideo(bitmap, t)) : null;
      const p = pose ? compactPose(pose.detectForVideo(bitmap, t)) : null;
      const inferenceMs = performance.now() - start;
      post({ type: 'result', t: msg.t, face: f, pose: p, inferenceMs });
      bitmap.close();
      await tune(inferenceMs);
    } catch (e) {
      bitmap.close();
      post({ type: 'error', message: e instanceof Error ? e.message : String(e) });
    }
  }
};
