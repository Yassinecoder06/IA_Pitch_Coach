import { create } from 'zustand';
import type { SensorFrame } from '@/lib/sensors/session';

export interface LiveWord {
  w: string;
  start: number;
  end: number;
  prob: number;
}

export interface LiveFiller {
  filler: string;
  start: number;
  end: number;
}

interface SensorState extends SensorFrame {
  words: LiveWord[];
  fillers: LiveFiller[];
  apply: (patch: SensorFrame) => void;
  addWords: (words: LiveWord[], fillers: LiveFiller[]) => void;
  reset: () => void;
}

const initial = { words: [], fillers: [], voice: undefined, gaze: undefined, posture: undefined, motion: undefined, vision: undefined, face: undefined, pose: undefined };

export const useSensors = create<SensorState>(set => ({
  ...initial,
  apply: patch => set(patch),
  addWords: (words, fillers) => set(s => ({ words: [...s.words, ...words], fillers: [...s.fillers, ...fillers] })),
  reset: () => set(initial),
}));

/** Words per minute over the last `windowS` seconds of transcribed speech. */
export function liveWpm(words: LiveWord[], windowS = 30): number {
  if (words.length < 3) return 0;
  const end = words[words.length - 1].end;
  const from = Math.max(0, end - windowS);
  const inWindow = words.filter(w => w.end > from);
  const span = Math.max(5, end - Math.max(from, inWindow[0]?.start ?? from));
  return Math.round((inWindow.length / span) * 60);
}
