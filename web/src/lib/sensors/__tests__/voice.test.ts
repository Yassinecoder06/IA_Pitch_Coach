import { describe, expect, it } from 'vitest';
import { yin } from '../audio/yin';
import { VoiceTracker } from '../audio/voiceTracker';

const SR = 16000;

function tone(seconds: number, hz: (t: number) => number, amp = 0.3): Float32Array {
  const out = new Float32Array(Math.round(seconds * SR));
  let phase = 0;
  for (let i = 0; i < out.length; i++) {
    phase += (2 * Math.PI * hz(i / SR)) / SR;
    // A few harmonics so it looks more like a voice than a pure sine.
    out[i] = amp * (Math.sin(phase) + 0.5 * Math.sin(2 * phase) + 0.25 * Math.sin(3 * phase)) / 1.75;
  }
  return out;
}

const silence = (seconds: number) => new Float32Array(Math.round(seconds * SR));

describe('yin', () => {
  it.each([110, 165, 220, 330])('finds %d Hz within 1%', hz => {
    const est = yin(tone(0.04, () => hz), SR);
    expect(Math.abs(est.f0 - hz) / hz).toBeLessThan(0.01);
    expect(est.clarity).toBeGreaterThan(0.8);
  });

  it('reports silence and noise as unvoiced', () => {
    expect(yin(silence(0.04), SR).f0).toBe(0);
    const noise = new Float32Array(640).map(() => Math.random() * 0.2 - 0.1);
    expect(yin(noise, SR).f0).toBe(0);
  });
});

describe('VoiceTracker', () => {
  it('flags a flat voice as monotone', () => {
    const t = new VoiceTracker(SR);
    t.push(silence(1));
    const snap = t.push(tone(3, () => 200));
    expect(snap.speaking).toBe(true);
    expect(snap.pitchRangeSt).toBeLessThan(1);
    expect(snap.monotone).toBe(true);
  });

  it('measures the semitone range of an expressive voice', () => {
    const t = new VoiceTracker(SR);
    t.push(silence(1));
    // Glide 150 -> 300 Hz (one octave = 12 semitones) and back.
    const snap = t.push(tone(4, s => 150 * 2 ** (s < 2 ? s / 2 : (4 - s) / 2)));
    expect(snap.monotone).toBe(false);
    expect(snap.pitchRangeSt).toBeGreaterThan(8);
    expect(snap.pitchRangeSt).toBeLessThan(13);
  });

  it('counts pauses between speech and tracks the current silence', () => {
    const t = new VoiceTracker(SR);
    t.push(silence(1));
    t.push(tone(1, () => 180));
    t.push(silence(0.6));
    t.push(tone(1, () => 180));
    t.push(silence(0.1)); // too short to be a pause
    t.push(tone(0.5, () => 180));
    let snap = t.push(silence(0.8));
    expect(snap.pauseCount).toBe(1);
    expect(snap.speaking).toBe(false);
    expect(snap.silenceMs).toBeGreaterThanOrEqual(700);
    snap = t.push(tone(0.3, () => 180));
    expect(snap.pauseCount).toBe(2);
    expect(snap.silenceMs).toBe(0);
  });

  it('treats continuous speech as speech even without a quiet lead-in', () => {
    const t = new VoiceTracker(SR);
    const snap = t.push(tone(4, () => 200));
    expect(snap.speaking).toBe(true);
    expect(snap.f0).toBeGreaterThan(190);
  });

  it('accepts frames of any size', () => {
    const a = new VoiceTracker(SR);
    const b = new VoiceTracker(SR);
    const signal = tone(2, () => 220);
    a.push(signal);
    for (let i = 0; i < signal.length; i += 1600) b.push(signal.subarray(i, i + 1600));
    expect(b.snapshot.f0).toBeCloseTo(a.snapshot.f0, 3);
  });
});
