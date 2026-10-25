import { startPcmRecorder, PCM_SAMPLE_RATE, type PcmRecorder } from '@/audio/pcmRecorder';
import { VoiceTracker, type VoiceSnapshot } from './audio/voiceTracker';
import { GazeTracker, type GazeReading } from './vision/gaze';
import { analysePosture, PostureSmoother, type PostureIssue, type PostureReading } from './vision/posture';
import { MotionTracker, type MotionReading } from './vision/motion';
import type { FaceFrame, PoseFrame } from './vision/types';
import type { VisionRequest, VisionResponse } from './vision/vision.worker';

export interface VisionStatus {
  state: 'off' | 'loading' | 'running' | 'error';
  delegate?: 'GPU' | 'CPU';
  fps: number;
  inferenceMs: number;
  error?: string;
}

export interface SensorFrame {
  voice?: VoiceSnapshot;
  gaze?: GazeReading;
  posture?: PostureReading;
  motion?: MotionReading;
  vision?: VisionStatus;
  face?: FaceFrame | null;
  pose?: PoseFrame | null;
}

/** One sample per second, stored with a rehearsal (numbers only, never pixels). */
export interface TimelineSample {
  t: number;
  speaking: boolean;
  levelDb: number;
  pitchRangeSt: number;
  monotone: boolean;
  face: boolean | null;
  gaze: number | null;
  gazeState: 'good' | 'weak' | 'lost' | null;
  posture: number | null;
  issues: PostureIssue[];
  fidget: number | null;
  sway: number | null;
  gesture: number | null;
}

export interface SensorSummary {
  duration_s: number;
  camera: boolean;
  eye_contact_pct: number | null;
  face_present_pct: number | null;
  posture_avg: number | null;
  posture_issue_seconds: Partial<Record<PostureIssue, number>>;
  fidget_avg: number | null;
  sway_avg: number | null;
  gesture_avg: number | null;
  pitch_range_avg_st: number | null;
  monotone_pct: number | null;
  energy_range_avg_db: number | null;
  pause_count: number;
  vision: { delegate?: string; avg_inference_ms: number; fps: number } | null;
  timeline: TimelineSample[];
}

export interface SensorOptions {
  camera: boolean;
  mic: boolean;
  video?: HTMLVideoElement | null;
  /** Receives every 100 ms PCM16 frame (e.g. to stream to the server). */
  onPcm?: (frame: ArrayBuffer) => void;
  onFrame?: (frame: SensorFrame) => void;
}

// Face and pose take turns, so each landmarker runs at half these frame rates (15 / 8 Hz).
const FAST_INTERVAL = 1000 / 30;
const SLOW_INTERVAL = 1000 / 16;
const mean = (xs: number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null);

/** Owns the camera, the mic, the vision worker and all trackers for one practice run. */
export class SensorSession {
  readonly gaze = new GazeTracker();
  private voice = new VoiceTracker(PCM_SAMPLE_RATE);
  private motion = new MotionTracker();
  private posture = new PostureSmoother();
  private recorder: PcmRecorder | null = null;
  private stream: MediaStream | null = null;
  private worker: Worker | null = null;
  private pumpTimer: ReturnType<typeof setInterval> | null = null;
  private sampleTimer: ReturnType<typeof setInterval> | null = null;
  private interval = FAST_INTERVAL;
  private inFlight = false;
  private sentAt = 0;
  private overBudget = 0;
  private inferenceEma = 0;
  private resultTimes: number[] = [];
  private startedAt = 0;
  private latest: SensorFrame = {};
  private timeline: TimelineSample[] = [];
  private visionStatus: VisionStatus = { state: 'off', fps: 0, inferenceMs: 0 };

  constructor(private readonly opts: SensorOptions) {}

  async start(): Promise<void> {
    this.startedAt = performance.now();
    if (this.opts.mic) {
      this.recorder = await startPcmRecorder(frame => {
        const snap = this.voice.pushInt16(new Int16Array(frame));
        this.emit({ voice: snap });
        this.opts.onPcm?.(frame);
      });
    }
    if (this.opts.camera && this.opts.video) await this.startCamera(this.opts.video);
    this.sampleTimer = setInterval(() => this.sample(), 1000);
  }

  private async startCamera(video: HTMLVideoElement) {
    this.setVision({ state: 'loading' });
    this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
    video.srcObject = this.stream;
    video.muted = true;
    await video.play().catch(() => undefined);

    this.worker = new Worker(new URL('./vision/vision.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<VisionResponse>) => this.onVision(e.data);
    this.worker.onerror = e => this.setVision({ state: 'error', error: e.message || 'Vision worker failed' });
    const base = new URL('/mediapipe', window.location.origin).href;
    this.worker.postMessage({ type: 'init', base, face: true, pose: true } satisfies VisionRequest);
  }

  private onVision(msg: VisionResponse) {
    if (msg.type === 'ready') {
      // Also sent again when the worker switches delegate after timing the first frames.
      const first = this.visionStatus.state !== 'running';
      this.setVision({ state: 'running', delegate: msg.delegate });
      this.inferenceEma = 0;
      this.overBudget = 0;
      if (first) this.schedulePump();
      return;
    }
    if (msg.type === 'error') {
      this.inFlight = false;
      if (this.visionStatus.state !== 'running') this.setVision({ state: 'error', error: msg.message });
      return;
    }
    this.inFlight = false;
    const now = performance.now();
    this.inferenceEma = this.inferenceEma ? 0.9 * this.inferenceEma + 0.1 * msg.inferenceMs : msg.inferenceMs;
    this.resultTimes.push(now);
    while (this.resultTimes.length && this.resultTimes[0] < now - 2000) this.resultTimes.shift();

    // Performance budget: inference runs in the worker, so it only has to keep up with the
    // frame interval. Sustained use of more than 80% of the interval drops to the slow rate.
    this.overBudget = msg.inferenceMs > this.interval * 0.8 ? this.overBudget + 1 : 0;
    if (this.overBudget >= 30 && this.interval === FAST_INTERVAL) {
      this.interval = SLOW_INTERVAL;
      this.schedulePump();
    }

    this.setVision({ fps: this.resultTimes.length / 2, inferenceMs: this.inferenceEma });
    if (msg.face !== undefined) this.emit({ gaze: this.gaze.update(msg.face, msg.t), face: msg.face });
    if (msg.pose !== undefined) {
      this.emit({ posture: this.posture.update(analysePosture(msg.pose)), motion: this.motion.update(msg.pose, msg.t), pose: msg.pose });
    }
  }

  private schedulePump() {
    if (this.pumpTimer) clearInterval(this.pumpTimer);
    this.pumpTimer = setInterval(() => void this.pump(), this.interval);
  }

  private async pump() {
    const video = this.opts.video;
    if (!this.worker || this.inFlight || !video || video.readyState < 2) return;
    this.inFlight = true; // backpressure: never queue frames behind a slow inference
    try {
      const bitmap = await createImageBitmap(video);
      this.sentAt = performance.now() - this.startedAt;
      this.worker.postMessage({ type: 'frame', bitmap, t: this.sentAt } satisfies VisionRequest, [bitmap]);
    } catch {
      this.inFlight = false;
    }
  }

  private setVision(patch: Partial<VisionStatus>) {
    this.visionStatus = { ...this.visionStatus, ...patch };
    this.emit({ vision: this.visionStatus });
  }

  private emit(patch: SensorFrame) {
    this.latest = { ...this.latest, ...patch };
    this.opts.onFrame?.(patch);
  }

  /** Seconds since start(). */
  get elapsed(): number {
    return (performance.now() - this.startedAt) / 1000;
  }

  private sample() {
    const { voice, gaze, posture, motion } = this.latest;
    const vision = this.visionStatus.state === 'running';
    this.timeline.push({
      t: Math.round(this.elapsed),
      speaking: !!voice?.speaking,
      levelDb: Math.round(voice?.levelDb ?? -100),
      pitchRangeSt: Math.round((voice?.pitchRangeSt ?? 0) * 10) / 10,
      monotone: !!voice?.monotone,
      face: vision && gaze ? gaze.facePresent : null,
      gaze: vision && gaze ? Math.round(gaze.score * 100) / 100 : null,
      gazeState: vision && gaze ? gaze.state : null,
      posture: vision && posture?.bodyPresent ? posture.score : null,
      issues: vision && posture ? posture.issues : [],
      fidget: vision && motion ? Math.round(motion.fidget * 100) / 100 : null,
      sway: vision && motion ? Math.round(motion.sway * 100) / 100 : null,
      gesture: vision && motion ? Math.round(motion.gestureEnergy * 100) / 100 : null,
    });
  }

  summary(): SensorSummary {
    const tl = this.timeline;
    const speaking = tl.filter(s => s.speaking);
    const tracked = tl.filter(s => s.face !== null);
    const issueSeconds: Partial<Record<PostureIssue, number>> = {};
    for (const s of tl) for (const i of s.issues) issueSeconds[i] = (issueSeconds[i] ?? 0) + 1;
    const camera = this.visionStatus.state === 'running';
    const nums = (pick: (s: TimelineSample) => number | null) => tl.map(pick).filter((x): x is number => x !== null);
    const round = (v: number | null, d = 1) => (v === null ? null : Math.round(v * 10 ** d) / 10 ** d);
    return {
      duration_s: Math.round(this.elapsed * 10) / 10,
      camera,
      eye_contact_pct: camera ? round(this.gaze.eyeContactShare * 100) : null,
      face_present_pct: camera && tracked.length ? round((tracked.filter(s => s.face).length / tracked.length) * 100) : null,
      posture_avg: round(mean(nums(s => s.posture))),
      posture_issue_seconds: issueSeconds,
      fidget_avg: round(mean(nums(s => s.fidget)), 2),
      sway_avg: round(mean(nums(s => s.sway)), 2),
      gesture_avg: round(mean(nums(s => s.gesture)), 2),
      pitch_range_avg_st: round(mean(speaking.map(s => s.pitchRangeSt).filter(x => x > 0))),
      monotone_pct: speaking.length ? round((speaking.filter(s => s.monotone).length / speaking.length) * 100) : null,
      energy_range_avg_db: round(this.voice.snapshot.energyRangeDb),
      pause_count: this.voice.snapshot.pauseCount,
      vision: camera ? { delegate: this.visionStatus.delegate, avg_inference_ms: round(this.inferenceEma) ?? 0, fps: this.visionStatus.fps } : null,
      timeline: tl,
    };
  }

  async stop(): Promise<SensorSummary> {
    if (this.sampleTimer) clearInterval(this.sampleTimer);
    if (this.pumpTimer) clearInterval(this.pumpTimer);
    this.sample();
    const summary = this.summary();
    await this.recorder?.stop();
    this.recorder = null;
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = null;
    this.worker?.terminate();
    this.worker = null;
    if (this.opts.video) this.opts.video.srcObject = null;
    return summary;
  }

  /** Point the camera at the speaker's normal position and record a "looking at the lens" baseline. */
  async calibrate(ms = 3000) {
    this.gaze.startCalibration();
    await new Promise(r => setTimeout(r, ms));
    return this.gaze.finishCalibration();
  }
}
