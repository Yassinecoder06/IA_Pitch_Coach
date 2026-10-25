import { describe, expect, it } from 'vitest';
import { EyeLock, FillerNinja, PaceRacer, PausePower, PitchSurfer, Statue } from '../games';
import type { VoiceSnapshot } from '@/lib/sensors/audio/voiceTracker';

const voice = (over: Partial<VoiceSnapshot>): VoiceSnapshot => ({
  f0: 180, pitchRangeSt: 5, monotone: false, levelDb: -20, energyRangeDb: 10, speaking: true, silenceMs: 0, pauseCount: 0, pitchTrail: [], ...over,
});

function run(game: { step: (i: any) => void; done: boolean }, frame: (t: number) => any, seconds: number, dt = 0.1) {
  for (let t = 0; t < seconds && !game.done; t += dt) game.step({ dt, ...frame(t) });
}

describe('Filler Ninja', () => {
  it('builds a combo on clean words and slices it on a filler', () => {
    const g = new FillerNinja('medium');
    g.step({ dt: 1, newWords: 12 });
    expect(g.state.combo).toBe(12);
    const before = g.score;
    g.step({ dt: 1, newWords: 3, newFillers: 1 });
    expect(g.state.combo).toBe(0);
    expect(g.score).toBeLessThan(before + 20 + 1); // 2 clean words then the penalty
    expect(g.stats()).toMatchObject({ fillers: 1, words: 15, max_combo: 14 });
  });

  it('caps a long frame gap so a hidden tab does not finish the game', () => {
    const g = new FillerNinja('easy');
    g.step({ dt: 60 });
    expect(g.elapsed).toBe(0.5);
  });
});

describe('Pitch Surfer', () => {
  it('rewards an expressive voice and wipes out a monotone one', () => {
    const lively = new PitchSurfer('medium');
    run(lively, () => ({ voice: voice({ pitchRangeSt: 9 }) }), 20);
    const flat = new PitchSurfer('medium');
    run(flat, () => ({ voice: voice({ pitchRangeSt: 1, monotone: true }) }), 20);
    expect(lively.score).toBeGreaterThan(flat.score);
    expect(flat.state.wipeouts).toBeGreaterThan(0);
    expect(lively.state.wipeouts).toBe(0);
  });
});

describe('Eye Lock', () => {
  it('fills the ring with steady eye contact and drains it when looking away', () => {
    const g = new EyeLock('medium'); // 15%/s
    run(g, () => ({ gaze: { score: 0.9, state: 'good', facePresent: true, yaw: 0, pitch: 0 } }), 7);
    expect(g.state.locks).toBe(1);
    run(g, () => ({ gaze: { score: 0, state: 'lost', facePresent: false, yaw: 0, pitch: 0 } }), 2);
    expect(g.state.charge).toBe(0);
    expect(g.stats().eye_contact_pct).toBeGreaterThan(70);
  });
});

describe('Stage Presence', () => {
  const posture = (score: number) => ({ score, issues: [], shoulderTiltDeg: 0, neckRatio: 0.5, bodyPresent: true });
  it('scores steadiness and counts breaks', () => {
    const g = new Statue('medium');
    run(g, () => ({ posture: posture(95), motion: { fidget: 0.1, sway: 0.1, gestureEnergy: 0 } }), 10);
    const steadyScore = g.score;
    run(g, () => ({ posture: posture(95), motion: { fidget: 0.9, sway: 0.1, gestureEnergy: 0 } }), 2);
    expect(g.score).toBeLessThan(steadyScore);
    expect(g.state.breaks).toBe(1);
    expect(g.stats().best_streak).toBe(10);
  });
});

describe('Pace Racer', () => {
  it('pays most inside the band and nothing on the redline', () => {
    const g = new PaceRacer('medium');
    run(g, () => ({ wpm: 145 }), 10);
    expect(g.state.inBand).toBeCloseTo(10, 0);
    const s = g.score;
    run(g, () => ({ wpm: 230 }), 5);
    expect(g.score).toBe(s);
    expect(g.state.redline).toBeGreaterThan(4);
  });
});

describe('Pause Power', () => {
  it('scores a pause of the right length after the cue and misses a rushed one', () => {
    const g = new PausePower('medium', () => 0); // next cue after exactly 4 s of speech
    run(g, () => ({ voice: voice({ speaking: true }) }), 6.05); // first cue at 6 s
    expect(g.state.phase).toBe('cue');
    run(g, () => ({ voice: voice({ speaking: false }) }), 1.5);
    g.step({ dt: 0.1, voice: voice({ speaking: true }) });
    expect(g.state.hits).toBe(1);

    run(g, () => ({ voice: voice({ speaking: true }) }), 4.1); // next cue
    run(g, () => ({ voice: voice({ speaking: false }) }), 0.3); // too short
    g.step({ dt: 0.1, voice: voice({ speaking: true }) });
    expect(g.state.misses).toBe(1);
    expect(g.score).toBe(100);
  });
});
