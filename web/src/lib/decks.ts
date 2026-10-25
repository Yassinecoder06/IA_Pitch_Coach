import { request } from './api';

export interface Slide {
  idx: number;
  kind: string | null;
  title: string;
  bullets: string[];
  notes: string;
  text: string;
  has_image: boolean;
  time_budget_sec: number | null;
}

export type SlideInput = Omit<Slide, 'idx' | 'has_image'>;

export interface SlideCritique {
  slide_number: number;
  density: 'ok' | 'too_dense' | 'too_sparse';
  issues: string[];
  suggestion: string;
}

export interface DeckCritique {
  overall: string;
  missing_sections: string[];
  slides: SlideCritique[];
  density: { slide_number: number; words: number; bullets: number; dense: boolean }[];
  model?: string;
}

export interface Deck {
  id: string;
  title: string;
  source: 'outline' | 'pdf' | 'pptx' | 'ai';
  brief: Record<string, unknown> | null;
  critique: DeckCritique | null;
  created_at: string;
  updated_at: string;
  slides: Slide[];
  slide_count: number;
}

export type DeckSummary = Omit<Deck, 'slides'>;

export interface DeckBrief {
  company: string;
  one_liner: string;
  audience: string;
  stage: string;
  minutes: number;
  details: string;
}

const json = (body: unknown) => ({ body: JSON.stringify(body) });

export const decksApi = {
  list: () => request<{ decks: DeckSummary[] }>('/decks'),
  get: (id: string) => request<Deck>(`/decks/${id}`),
  create: (title: string, source: Deck['source'], slides: Partial<SlideInput>[]) =>
    request<Deck>('/decks', { method: 'POST', ...json({ title, source, slides }) }),
  generate: (brief: DeckBrief) => request<Deck>('/decks/generate', { method: 'POST', ...json(brief) }),
  rename: (id: string, title: string) => request<Deck>(`/decks/${id}`, { method: 'PATCH', ...json({ title }) }),
  replaceSlides: (id: string, slides: Partial<SlideInput>[]) =>
    request<Deck>(`/decks/${id}/slides`, { method: 'PUT', ...json({ slides }) }),
  remove: (id: string) => request<{ ok: boolean }>(`/decks/${id}`, { method: 'DELETE' }),
  critique: (id: string) => request<DeckCritique>(`/decks/${id}/critique`, { method: 'POST' }),
  uploadImage: (id: string, idx: number, blob: Blob) =>
    request<{ ok: boolean }>(`/decks/${id}/slides/${idx}/image`, {
      method: 'PUT',
      body: blob,
      headers: { 'Content-Type': blob.type },
    }),
  convertPptx: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return request<{ pdf_url: string; slides: { title: string; text: string; notes: string }[] }>('/decks/convert-pptx', {
      method: 'POST',
      body: form,
    });
  },
  imageUrl: (id: string, idx: number, version = '') => `/api/decks/${id}/slides/${idx}/image${version ? `?v=${version}` : ''}`,
  exportUrl: (id: string) => `/api/decks/${id}/export.pptx`,
};
