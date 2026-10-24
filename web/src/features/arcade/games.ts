/**
 * Training-arcade game logic. Pure state machines: the UI feeds them sensor readings
 * every animation frame and renders their state, tests feed them scripted inputs.
 */
import type { VoiceSnapshot } from '@/lib/sensors/audio/voiceTracker';
import type { GazeReading } from '@/lib/sensors/vision/gaze';
import type { PostureReading } from '@/lib/sensors/vision/posture';
import type { MotionReading } from '@/lib/sensors/vision/motion';

export type GameId = 'filler_ninja' | 'pitch_surfer' | 'eye_lock' | 'statue' | 'pace_racer' | 'pause_power';
export type Difficulty = 'easy' | 'medium' | 'hard';

export interface FrameInput {
  dt: number; // seconds since the previous frame
  voice?: VoiceSnapshot;
  gaze?: GazeReading;
  posture?: PostureReading;
  motion?: MotionReading;
  wpm?: number;
  newWords?: number; // words transcribed since the previous frame
  newFillers?: number;
}

export interface Game<S> {
  readonly duration: number; // seconds
  state: S;
  elapsed: number;
  step(input: FrameInput): void;
  get score(): number;
  get done(): boolean;
  stats(): Record<string, number>;
}

abstract class Timed<S> implements Game<S> {
  elapsed = 0;
  constructor(readonly duration: number, public state: S) {}
  step(input: FrameInput) {
    if (this.done) return;
    const dt = Math.min(input.dt, 0.5); // a hidden tab must not award points for the gap
    this.elapsed = Math.min(this.duration, this.elapsed + dt);
    this.update({ ...input, dt });
  }
  protected abstract update(input: FrameInput): void;
  abstract get score(): number;
  abstract stats(): Record<string, number>;
  get done() {
    return this.elapsed >= this.duration;
  }
}

const pick = <T,>(d: Difficulty, easy: T, medium: T, hard: T) => (d === 'easy' ? easy : d === 'medium' ? medium : hard);
const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));

/* ---------- Filler Ninja: every clean word builds a combo; a filler slices it ---------- */
export class FillerNinja extends Timed<{ points: number; combo: number; maxCombo: number; fillers: number; words: number; slashed: number }> {
  private penalty: number;
  constructor(d: Difficulty) {
    super(pick(d, 45, 60, 90), { points: 0, combo: 0, maxCombo: 0, fillers: 0, words: 0, slashed: 0 });
    this.penalty = pick(d, 15, 30, 50);
  }
  protected update({ newWords = 0, newFillers = 0 }: FrameInput) {
    const s = this.state;
    const clean = Math.max(0, newWords - newFillers);
    for (let i = 0; i < clean; i++) {
      s.combo += 1;
      s.points += 10 * (1 + Math.floor(s.combo / 10)); // x2 after 10 clean words, x3 after 20...
    }
    s.words += newWords;
    s.maxCombo = Math.max(s.maxCombo, s.combo); // before a filler resets the combo
    if (newFillers > 0) {
      s.fillers += newFillers;
      s.slashed += 1;
      s.combo = 0;
      s.points = Math.max(0, s.points - this.penalty * newFillers);
    }
  }
  get score() {
    return this.state.points;
  }
  stats() {
    const { fillers, words, maxCombo } = this.state;
    return { fillers, words, max_combo: maxCombo, fillers_per_100_words: words ? Math.round((fillers / words) * 1000) / 10 : 0 };
  }
}

/* ---------- Pitch Surfer: vocal variety keeps you on the wave; monotone sinks you ---------- */
export class PitchSurfer extends Timed<{ altitude: number; onWave: number; wipeouts: number; points: number }> {
  private target: number;
  constructor(d: Difficulty) {
    super(pick(d, 45, 60, 75), { altitude: 0.5, onWave: 0, wipeouts: 0, points: 0 });
    this.target = pick(d, 3, 5, 7); // semitones of range that hold altitude steady
  }
  protected update({ dt, voice }: FrameInput) {
    const s = this.state;
    if (voice?.speaking && voice.pitchRangeSt > 0) {
      s.altitude += clamp((voice.pitchRangeSt - this.target) / this.target, -1, 1) * 0.4 * dt;
    } else {
      s.altitude -= 0.08 * dt; // silence slowly loses the wave
    }
    if (s.altitude <= 0) {
      s.wipeouts += 1;
      s.altitude = 0.35; // back on the board
    }
    s.altitude = clamp(s.altitude);
    if (s.altitude > 0.3 && voice?.speaking) {
      s.onWave += dt;
      s.points += dt * 10 * (1 + s.altitude);
    }
  }
  get score() {
    return Math.round(this.state.points);
  }
  stats() {
    return { seconds_on_wave: Math.round(this.state.onWave), wipeouts: this.state.wipeouts };
  }
}

/* ---------- Eye Lock: the ring charges while you hold the lens, drains when you look away ---------- */
export class EyeLock extends Timed<{ charge: number; locks: number; goodSeconds: number; tracked: number }> {
  private rate: number;
  constructor(d: Difficulty) {
    super(pick(d, 30, 45, 60), { charge: 0, locks: 0, goodSeconds: 0, tracked: 0 });
    this.rate = pick(d, 25, 15, 10); // % per second while eye contact is good
  }
  protected update({ dt, gaze }: FrameInput) {
    const s = this.state;
    s.tracked += dt;
    const state = gaze?.facePresent ? gaze.state : 'lost';
    if (state === 'good') {
      s.goodSeconds += dt;
      s.charge += this.rate * dt;
    } else {
      s.charge -= (state === 'weak' ? 10 : 30) * dt;
    }
    if (s.charge >= 100) {
      s.locks += 1;
      s.charge = 0; // a full ring is banked as a "lock"
    }
    s.charge = clamp(s.charge, 0, 100);
  }
  get score() {
    return Math.round(this.state.locks * 100 + this.state.goodSeconds * 10);
  }
  stats() {
    const { locks, goodSeconds, tracked } = this.state;
    return { locks, eye_contact_pct: tracked ? Math.round((goodSeconds / tracked) * 100) : 0 };
  }
}

/* ---------- Stage Presence ("Statue"): open, steady posture earns points; fidgets cost them ---------- */
export class Statue extends Timed<{ points: number; steady: number; streak: number; bestStreak: number; breaks: number }> {
  private limit: number;
  private wasSteady = true;
  constructor(d: Difficulty) {
    super(pick(d, 30, 45, 60), { points: 0, steady: 0, streak: 0, bestStreak: 0, breaks: 0 });
    this.limit = pick(d, 0.5, 0.35, 0.25);
  }
  protected update({ dt, posture, motion }: FrameInput) {
    const s = this.state;
    const steady =
      !!posture?.bodyPresent &&
      posture.score >= 80 &&
      (motion?.fidget ?? 0) < this.limit &&
      (motion?.sway ?? 0) < this.limit;
    if (steady) {
      s.steady += dt;
      s.streak += dt;
      s.bestStreak = Math.max(s.bestStreak, s.streak);
      s.points += dt * 10 * (1 + Math.min(2, s.streak / 10));
    } else {
      if (this.wasSteady) s.breaks += 1;
      s.streak = 0;
      s.points = Math.max(0, s.points - dt * 5);
    }
    this.wasSteady = steady;
  }
  get score() {
    return Math.round(this.state.points);
  }
  stats() {
    return { steady_seconds: Math.round(this.state.steady), best_streak: Math.round(this.state.bestStreak), breaks: this.state.breaks };
  }
}

/* ---------- Pace Racer: stay in the green band ---------- */
export class PaceRacer extends Timed<{ points: number; inBand: number; redline: number; wpm: number }> {
  readonly band: [number, number];
  constructor(d: Difficulty) {
    super(pick(d, 45, 60, 90), { points: 0, inBand: 0, redline: 0, wpm: 0 });
    this.band = pick(d, [120, 170], [130, 160], [138, 152]) as [number, number];
  }
  protected update({ dt, wpm = 0 }: FrameInput) {
    const s = this.state;
    s.wpm = wpm;
    if (!wpm) return; // no pace until a few words are transcribed
    const [lo, hi] = this.band;
    if (wpm >= lo && wpm <= hi) {
      s.inBand += dt;
      s.points += 10 * dt;
    } else if (wpm >= lo - 15 && wpm <= hi + 15) {
      s.points += 4 * dt;
    } else {
      s.redline += dt;
    }
  }
  get score() {
    return Math.round(this.state.points);
  }
  stats() {
    return { seconds_in_band: Math.round(this.state.inBand), seconds_redline: Math.round(this.state.redline) };
  }
}

/* ---------- Pause Power: hold a deliberate pause on cue, then carry on ---------- */
type PausePhase = 'talk' | 'cue' | 'pausing' | 'resume';
export class PausePower extends Timed<{ phase: PausePhase; cues: number; hits: number; misses: number; points: number; pauseMs: number; untilCue: number }> {
  readonly window: [number, number]; // accepted pause length in seconds
  private cueTimeout = 3;
  private sinceCue = 0;
  constructor(d: Difficulty, private readonly random: () => number = Math.random) {
    super(pick(d, 45, 60, 75), { phase: 'talk', cues: 0, hits: 0, misses: 0, points: 0, pauseMs: 0, untilCue: 6 });
    this.window = pick(d, [0.8, 3], [1, 2.5], [1.5, 2]) as [number, number];
  }
  protected update({ dt, voice }: FrameInput) {
    const s = this.state;
    const speaking = !!voice?.speaking;
    switch (s.phase) {
      case 'talk':
        if (speaking) s.untilCue -= dt; // cues only interrupt actual speech
        if (s.untilCue <= 0) {
          s.phase = 'cue';
          s.cues += 1;
          this.sinceCue = 0;
        }
        break;
      case 'cue':
        this.sinceCue += dt;
        if (!speaking) {
          s.phase = 'pausing';
          s.pauseMs = 0;
        } else if (this.sinceCue > this.cueTimeout) {
          this.miss();
        }
        break;
      case 'pausing':
        if (!speaking) {
          s.pauseMs += dt * 1000;
          if (s.pauseMs / 1000 > this.window[1] + 1) this.miss(); // lost the thread
        } else {
          const len = s.pauseMs / 1000;
          if (len >= this.window[0] && len <= this.window[1]) {
            s.hits += 1;
            s.points += 100;
            s.phase = 'resume';
            this.next();
          } else this.miss();
        }
        break;
      case 'resume':
        s.phase = 'talk';
        break;
    }
  }
  private miss() {
    this.state.misses += 1;
    this.state.phase = 'talk';
    this.next();
  }
  private next() {
    this.state.untilCue = 4 + this.random() * 5;
  }
  get score() {
    return this.state.points;
  }
  stats() {
    return { cues: this.state.cues, hits: this.state.hits, misses: this.state.misses };
  }
}

export interface GameInfo {
  id: GameId;
  title: string;
  skill: string;
  how: string;
  needs: { mic: boolean; camera: boolean; captions: boolean };
  create: (d: Difficulty) => Game<any>;
}

export const GAMES: Record<GameId, GameInfo> = {
  filler_ninja: {
    id: 'filler_ninja', title: 'Filler Ninja', skill: 'Fillers',
    how: 'Talk about the prompt. Every clean word builds your combo; each "um", "like" or "you know" slices it.',
    needs: { mic: true, camera: false, captions: true }, create: d => new FillerNinja(d),
  },
  pitch_surfer: {
    id: 'pitch_surfer', title: 'Pitch Surfer', skill: 'Vocal variety',
    how: 'Ride the wave with an expressive voice. A flat, monotone voice sinks your surfer.',
    needs: { mic: true, camera: false, captions: false }, create: d => new PitchSurfer(d),
  },
  eye_lock: {
    id: 'eye_lock', title: 'Eye Lock', skill: 'Eye contact',
    how: 'Hold your gaze on the camera lens to fill the ring. Looking away drains it.',
    needs: { mic: false, camera: true, captions: false }, create: d => new EyeLock(d),
  },
  statue: {
    id: 'statue', title: 'Stage Presence', skill: 'Composure',
    how: 'Stand or sit tall with open arms. Hold still between gestures: fidgeting, swaying and crossed arms cost points.',
    needs: { mic: false, camera: true, captions: false }, create: d => new Statue(d),
  },
  pace_racer: {
    id: 'pace_racer', title: 'Pace Racer', skill: 'Pace',
    how: 'Talk on the prompt and keep your speed inside the green band.',
    needs: { mic: true, camera: false, captions: true }, create: d => new PaceRacer(d),
  },
  pause_power: {
    id: 'pause_power', title: 'Pause Power', skill: 'Strategic pauses',
    how: 'Keep talking. When PAUSE flashes, stop for a deliberate beat, then carry on.',
    needs: { mic: true, camera: false, captions: false }, create: d => new PausePower(d),
  },
};

export const PROMPTS = [
  'Explain your product to a smart 12-year-old.',
  'Why is now the right time for your company?',
  'Describe the moment a customer realised they needed you.',
  'What would you do with an extra million dollars tomorrow?',
  'Pitch your favourite city as if it were a startup.',
  'What is the biggest myth in your industry?',
  'Tell the story of your first paying customer.',
];
