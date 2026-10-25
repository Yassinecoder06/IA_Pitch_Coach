export interface Profile {
  id: string;
  name: string;
  color: string;
  has_pin: boolean;
  created_at: string | null;
  last_used_at: string | null;
}

export interface OwnerStatus {
  configured: boolean;
  authenticated: boolean;
  profile: Profile | null;
}

export interface Session {
  id: string;
  profile_id: string;
  title: string;
  current_mode: string;
  context_md?: string;
  created_at: string;
  updated_at: string;
}

export interface DimensionScore {
  score: number;
  feedback?: string;
}

export interface StructureChecklist {
  hook: boolean;
  problem: boolean;
  solution: boolean;
  market: boolean;
  traction: boolean;
  ask: boolean;
}

export interface Scorecard {
  overall: number;
  clarity: DimensionScore;
  structure: DimensionScore;
  persuasion: DimensionScore;
  delivery: DimensionScore;
  confidence: DimensionScore;
  relevance: DimensionScore;
  checklist?: StructureChecklist;
  strengths?: string[];
  top_fixes?: string[];
  rewrite?: string | null;
}

export interface TranscriptWord {
  w: string;
  start: number;
  end: number;
  prob: number;
}

export interface SpeechMetrics {
  words_per_minute?: number;
  pause_frequency?: number;
  pause_duration?: number;
  energy_variation?: number;
  rhythm_score?: number;
  filler_count?: number;
}

export interface Message {
  id: string;
  session_id: string;
  role: 'user' | 'assistant';
  content: string;
  transcript?: string;
  scores?: Scorecard | null;
  metrics?: SpeechMetrics | null;
  audio_url?: string | null;
  created_at: string;
}

export interface CoachingMode {
  id: string;
  name: string;
  description: string;
  icon: string;
  produces_scorecard: boolean;
  default_tts: boolean;
}

export interface ModelInfo {
  id: string;
  label: string;
  capabilities: string[];
}

export interface Provider {
  id: string;
  kind: string;
  name: string;
  base_url?: string | null;
  has_key?: boolean;
  key_hint?: string | null;
  enabled: boolean | number;
  models: ModelInfo[];
  last_fetched_at?: string | null;
  last_error?: string | null;
}

export interface CatalogItem {
  ref: string;
  providerId: string;
  providerName: string;
  kind: string;
  model: string;
  label: string;
  capabilities: string[];
}

export type AiTask = 'chat' | 'judge' | 'deck_gen' | 'qa';

export interface AiPreferences {
  chat_models: string[];
  fallback_models: string[];
  task_models: Partial<Record<AiTask, string>>;
  tts_voice: string;
}

export interface AiProvidersState {
  providers: Provider[];
  preferences: AiPreferences;
  catalog: CatalogItem[];
  kinds: {
    kind: string;
    label: string;
    needsKey: boolean;
    needsBaseUrl: boolean;
  }[];
  tasks: AiTask[];
}
