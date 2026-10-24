import { rms, semitones, toDb, yin } from './yin';

export interface VoiceSnapshot {
  /** Current pitch in Hz (0 when silent or unvoiced). */
  f0: number;
  /** Pitch spread over the last few seconds of speech, p10..p90 in semitones. */
  pitchRangeSt: number;
  /** True when enough speech was heard and the pitch barely moves. */
  monotone: boolean;
  /** Loudness of the latest hop in dBFS. */
  levelDb: number;
  /** Loudness spread while speaking (p25..p90 in dB): vocal energy variation. */
  energyRangeDb: number;
  speaking: boolean;
  /** Length of the current silence in ms (0 while speaking). */
  silenceMs: number;
  /** Pauses of 0.3 s or longer between speech, since the tracker started. */
  pauseCount: number;
  /** Recent pitch samples for sparklines (Hz, 0 = unvoiced). */
  pitchTrail: number[];
}

const HOP_MS = 20;
const WINDOW_MS = 40;
const PITCH_WINDOW_S = 6;
const MIN_VOICED_FOR_RANGE = 25; // ~0.5 s of voiced hops
export const MONOTONE_SEMITONES = 3;
const PAUSE_MIN_MS = 300;
const TRAIL = 150; // 3 s at 20 ms hops

function percentile(sorted: number[], p: number): number {
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))];
}

/**
 * Rolling voice analysis over a stream of mono PCM frames.
 * Feed it frames of any size; it analyses 40 ms windows every 20 ms.
 */
export class VoiceTracker {
  private readonly hop: number;
  private readonly win: number;
  private buffer: Float32Array;
  private filled = 0;
  private voiced: { t: number; f0: number }[] = [];
  private levels: { t: number; db: number }[] = [];
  private noiseFloorDb = -60;
  private floorSamples: number[] = [];
  private hops = 0;
  private silenceHops = 0;
  private heardSpeech = false;
  private trail: number[] = [];
  private snapshotValue: VoiceSnapshot = {
    f0: 0,
    pitchRangeSt: 0,
    monotone: false,
    levelDb: -100,
    energyRangeDb: 0,
    speaking: false,
    silenceMs: 0,
    pauseCount: 0,
    pitchTrail: [],
  };

  constructor(readonly sampleRate = 16000) {
    this.hop = Math.round((sampleRate * HOP_MS) / 1000);
    this.win = Math.round((sampleRate * WINDOW_MS) / 1000);
    this.buffer = new Float32Array(this.win);
  }

  pushInt16(frame: Int16Array): VoiceSnapshot {
    const f = new Float32Array(frame.length);
    for (let i = 0; i < frame.length; i++) f[i] = frame[i] / 32768;
    return this.push(f);
  }

  push(samples: Float32Array): VoiceSnapshot {
    let offset = 0;
    while (offset < samples.length) {
      const take = Math.min(this.hop, samples.length - offset, this.win - this.filled);
      this.buffer.set(samples.subarray(offset, offset + take), this.filled);
      this.filled += take;
      offset += take;
      if (this.filled === this.win) {
        this.analyseWindow();
        // Slide by one hop
        this.buffer.copyWithin(0, this.hop);
        this.filled = this.win - this.hop;
      }
    }
    return this.snapshotValue;
  }

  get snapshot(): VoiceSnapshot {
    return this.snapshotValue;
  }

  private analyseWindow() {
    const t = (this.hops++ * HOP_MS) / 1000;
    const level = toDb(rms(this.buffer));

    // Adaptive noise floor: 10th percentile of recent levels.
    this.floorSamples.push(level);
    if (this.floorSamples.length > 500) this.floorSamples.shift();
    if (this.hops % 25 === 0) {
      this.noiseFloorDb = percentile([...this.floorSamples].sort((a, b) => a - b), 0.1);
    }
    // 10 dB over the noise floor, clamped to [-50, -35] dBFS: before any silence has been
    // heard the "floor" is the voice itself, which must not turn speech into silence.
    const speaking = level > Math.min(-35, Math.max(-50, this.noiseFloorDb + 10));

    let f0 = 0;
    if (speaking) {
      const est = yin(this.buffer, this.sampleRate);
      if (est.f0 > 0 && est.clarity > 0.5) f0 = est.f0;
      this.levels.push({ t, db: level });
    }
    if (f0 > 0) this.voiced.push({ t, f0 });

    const cutoff = t - PITCH_WINDOW_S;
    while (this.voiced.length && this.voiced[0].t < cutoff) this.voiced.shift();
    while (this.levels.length && this.levels[0].t < cutoff) this.levels.shift();

    // Pauses: silence of >= PAUSE_MIN_MS that ends when speech resumes.
    let pauseCount = this.snapshotValue.pauseCount;
    if (speaking) {
      if (this.heardSpeech && this.silenceHops * HOP_MS >= PAUSE_MIN_MS) pauseCount++;
      this.heardSpeech = true;
      this.silenceHops = 0;
    } else {
      this.silenceHops++;
    }

    let pitchRangeSt = 0;
    if (this.voiced.length >= MIN_VOICED_FOR_RANGE) {
      const sorted = this.voiced.map(v => v.f0).sort((a, b) => a - b);
      pitchRangeSt = semitones(percentile(sorted, 0.9), percentile(sorted, 0.1));
    }
    let energyRangeDb = 0;
    if (this.levels.length >= MIN_VOICED_FOR_RANGE) {
      const sorted = this.levels.map(l => l.db).sort((a, b) => a - b);
      energyRangeDb = percentile(sorted, 0.9) - percentile(sorted, 0.25);
    }

    this.trail.push(f0);
    if (this.trail.length > TRAIL) this.trail.shift();

    this.snapshotValue = {
      f0,
      pitchRangeSt,
      monotone: this.voiced.length >= MIN_VOICED_FOR_RANGE && pitchRangeSt < MONOTONE_SEMITONES,
      levelDb: level,
      energyRangeDb,
      speaking,
      silenceMs: speaking ? 0 : this.silenceHops * HOP_MS,
      pauseCount,
      pitchTrail: this.trail.slice(),
    };
  }
}
