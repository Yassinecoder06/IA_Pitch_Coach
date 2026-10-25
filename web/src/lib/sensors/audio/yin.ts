/**
 * YIN fundamental-frequency estimator (de Cheveigné & Kawahara, 2002).
 * Pure function over a mono Float32 frame; returns 0 for unvoiced frames.
 */
export interface PitchEstimate {
  f0: number; // Hz, 0 when unvoiced
  clarity: number; // 0..1, 1 - normalised difference at the chosen lag
}

export function yin(
  frame: Float32Array,
  sampleRate: number,
  minHz = 75,
  maxHz = 400,
  threshold = 0.15,
): PitchEstimate {
  const minTau = Math.max(2, Math.floor(sampleRate / maxHz));
  const maxTau = Math.min(Math.floor(frame.length / 2), Math.ceil(sampleRate / minHz));
  if (maxTau <= minTau) return { f0: 0, clarity: 0 };

  const window = frame.length - maxTau;
  const diff = new Float32Array(maxTau + 1);
  for (let tau = 1; tau <= maxTau; tau++) {
    let sum = 0;
    for (let j = 0; j < window; j++) {
      const d = frame[j] - frame[j + tau];
      sum += d * d;
    }
    diff[tau] = sum;
  }

  // Cumulative mean normalised difference
  const cmnd = new Float32Array(maxTau + 1);
  cmnd[0] = 1;
  let running = 0;
  for (let tau = 1; tau <= maxTau; tau++) {
    running += diff[tau];
    cmnd[tau] = running > 0 ? (diff[tau] * tau) / running : 1;
  }

  let tau = -1;
  for (let t = minTau; t <= maxTau; t++) {
    if (cmnd[t] < threshold) {
      while (t + 1 <= maxTau && cmnd[t + 1] < cmnd[t]) t++;
      tau = t;
      break;
    }
  }
  if (tau === -1) return { f0: 0, clarity: 0 };

  // Parabolic interpolation around the minimum for sub-sample precision
  let betterTau = tau;
  if (tau > 1 && tau < maxTau) {
    const s0 = cmnd[tau - 1];
    const s1 = cmnd[tau];
    const s2 = cmnd[tau + 1];
    const denom = 2 * (2 * s1 - s2 - s0);
    if (denom !== 0) betterTau = tau + (s2 - s0) / denom;
  }
  return { f0: sampleRate / betterTau, clarity: Math.max(0, 1 - cmnd[tau]) };
}

export function rms(frame: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
  return Math.sqrt(sum / Math.max(1, frame.length));
}

export const toDb = (value: number) => 20 * Math.log10(Math.max(value, 1e-6));
export const semitones = (hz: number, refHz: number) => 12 * Math.log2(hz / refHz);
