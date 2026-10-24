import { request } from './api';
import type { Scorecard, TranscriptWord } from './types';
import type { SensorSummary } from './sensors/session';

export type MetricKey =
  | 'pace' | 'fillers' | 'pauses' | 'vocal_variety' | 'energy' | 'eye_contact'
  | 'posture' | 'composure' | 'language' | 'clarity' | 'timing';

export type Scores = Partial<Record<MetricKey, number | null>> & { delivery?: number | null; content?: number | null };

export interface Cue {
  id: string;
  kind: string;
  severity: 'critical' | 'warn' | 'info';
  slide_idx: number | null;
  t: number | null;
  message: string;
  evidence: string | null;
  value: unknown;
}

export interface SegmentStats {
  duration: number;
  budget: number | null;
  words: number;
  wpm: number;
  fillers: number;
  eye_contact_pct: number | null;
  gaze_avg: number | null;
  text: string;
}

export interface Judgement {
  scorecard: Scorecard;
  summary: string;
  strengths: string[];
  priorities: { title: string; detail: string; evidence: string[] }[];
  slide_feedback: { slide_idx: number; feedback: string; suggested_line: string | null }[];
  rewrite: { original: string; improved: string } | null;
  model?: string;
}

export interface LanguageItem {
  kind: string;
  text: string;
  word_index: number;
  t: number;
  tip: string;
}

export interface Analysis {
  duration: number;
  word_count: number;
  pace: { wpm: number; articulation_wpm: number; windows: { start: number; end: number; wpm: number }[]; variability: number };
  pauses: { count: number; per_minute: number; strategic: number; hesitation: number; long: number; longest: number; strategic_share: number; items: { start: number; end: number; duration: number; kind: string }[] };
  fillers: { count: number; per_minute: number; by_word: Record<string, number>; items: { filler: string; start: number; end: number; word_index: number }[] };
  prosody: { median_hz: number; range_st: number; monotone: boolean; dynamics_db: number; windows: { start: number; t?: number; range_st: number }[] };
  upspeak: { sentences_checked: number; rising: number; share: number; items: { word_index: number; t: number; rise_st: number }[] };
  language: { counts: Record<string, number>; hedges_per_minute: number; items: LanguageItem[] };
  clarity: { avg_confidence: number; unclear_count: number; unclear: { word_index: number; text: string; t: number; prob: number }[]; key_terms_missing: string[]; key_terms_unclear: string[] };
  scores: Scores;
  delivery_score: number | null;
  target_seconds: number | null;
}

export interface RehearsalSummary {
  id: string;
  deck_id: string | null;
  title: string;
  status: 'ready' | 'recording' | 'processing' | 'done' | 'failed';
  progress: string | null;
  target_seconds: number | null;
  duration_sec: number | null;
  overall_score: number | null;
  scores: Scores | null;
  cue_ids: string[];
  error: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface Rehearsal extends RehearsalSummary {
  transcript: string | null;
  words: TranscriptWord[];
  analysis: Analysis | null;
  sensors: Partial<SensorSummary> & { slide_events?: { index: number; t: number }[] };
  cues: Cue[];
  judge: Judgement | null;
  segments: { slide_idx: number; start: number; end: number; stats: SegmentStats }[];
  has_audio: boolean;
}

export const METRIC_LABELS: Record<MetricKey, string> = {
  pace: 'Pace',
  fillers: 'Fillers',
  pauses: 'Pauses',
  vocal_variety: 'Vocal variety',
  energy: 'Vocal energy',
  eye_contact: 'Eye contact',
  posture: 'Posture',
  composure: 'Composure',
  language: 'Word choice',
  clarity: 'Clarity',
  timing: 'Timing',
};

export interface Comparison {
  a: Pick<RehearsalSummary, 'id' | 'title' | 'created_at' | 'deck_id' | 'overall_score'>;
  b: Pick<RehearsalSummary, 'id' | 'title' | 'created_at' | 'deck_id' | 'overall_score'>;
  scores: { metric: string; a: number | null; b: number | null; delta: number | null }[];
  facts: { metric: string; a: number | null; b: number | null; delta: number | null }[];
  cues: { id: string; status: 'resolved' | 'persisting' | 'new'; message: string }[];
  priorities: { a: string[]; b: string[] };
}

export interface Goal {
  id: string;
  metric: string;
  op: 'gte' | 'lte' | 'between';
  target: number;
  target_max: number | null;
  latest: number | null;
  recent_avg: number | null;
  status: 'met' | 'not_met' | 'no_data';
}

export interface PrivacySettings {
  keep_recordings: boolean;
  recording_retention_days: number | null;
}

export const SCORE_LABELS: Record<string, string> = { overall: 'Overall', delivery: 'Delivery', content: 'Content', ...METRIC_LABELS };

/** Raw measurements: label, unit, and which direction is better (null = a target band). */
export const FACT_INFO: Record<string, { label: string; unit: string; better: 'up' | 'down' | null }> = {
  duration_s: { label: 'Duration', unit: 's', better: null },
  wpm: { label: 'Pace', unit: 'wpm', better: null },
  fillers_per_min: { label: 'Fillers', unit: '/min', better: 'down' },
  pitch_range_st: { label: 'Pitch range', unit: 'st', better: 'up' },
  eye_contact_pct: { label: 'Eye contact', unit: '%', better: 'up' },
  posture_avg: { label: 'Posture', unit: '', better: 'up' },
  hedges_per_min: { label: 'Hedges', unit: '/min', better: 'down' },
};

export const rehearsalsApi = {
  create: (body: { title: string; deck_id?: string | null; target_seconds?: number | null }) =>
    request<RehearsalSummary>('/rehearsals', { method: 'POST', body: JSON.stringify(body) }),
  list: (deckId?: string) => request<{ rehearsals: RehearsalSummary[] }>(`/rehearsals${deckId ? `?deck_id=${deckId}` : ''}`),
  get: (id: string) => request<Rehearsal>(`/rehearsals/${id}`),
  remove: (id: string) => request<{ ok: boolean }>(`/rehearsals/${id}`, { method: 'DELETE' }),
  reanalyse: (id: string) => request<Rehearsal>(`/rehearsals/${id}/reanalyse`, { method: 'POST' }),
  startQa: (id: string) => request<{ session_id: string }>(`/rehearsals/${id}/qa`, { method: 'POST' }),
  audioUrl: (id: string) => `/api/rehearsals/${id}/audio`,
  deleteAudio: (id: string) => request<{ ok: boolean }>(`/rehearsals/${id}/audio`, { method: 'DELETE' }),
  compare: (a: string, b: string) => request<Comparison>(`/rehearsals/compare?a=${a}&b=${b}`),
  goals: () => request<{ goals: Goal[]; metrics: string[]; ops: string[] }>('/rehearsals/goals'),
  addGoal: (g: { metric: string; op: string; target: number; target_max?: number | null }) =>
    request<Goal>('/rehearsals/goals', { method: 'POST', body: JSON.stringify(g) }),
  deleteGoal: (id: string) => request<{ ok: boolean }>(`/rehearsals/goals/${id}`, { method: 'DELETE' }),
  privacy: () => request<PrivacySettings>('/rehearsals/privacy'),
  savePrivacy: (p: PrivacySettings) => request<PrivacySettings>('/rehearsals/privacy', { method: 'PUT', body: JSON.stringify(p) }),
};
