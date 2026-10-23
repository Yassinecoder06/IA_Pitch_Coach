import type { AiPreferences, AiProvidersState, CoachingMode, Message, OwnerStatus, Profile, Session } from './types';

const API_BASE = '/api';

export const AUTH_REQUIRED_EVENT = 'pitch-coach-auth-required';
export const PROFILE_REQUIRED_EVENT = 'pitch-coach-profile-required';

export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly code?: string) {
    super(message);
  }
}

function errorMessage(data: any, status: number): string {
  const detail = data?.detail;
  if (typeof detail === 'string') return detail;
  if (detail?.error) return detail.error;
  if (Array.isArray(detail)) return detail.map((d: any) => d.msg).join('; '); // FastAPI validation errors
  return data?.error || `Request failed with ${status}`;
}

export async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers || {});
  if (!headers.has('Content-Type') && options.body && typeof options.body === 'string') {
    headers.set('Content-Type', 'application/json');
  }

  // The session lives in an HttpOnly cookie; the browser attaches it automatically.
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
    credentials: 'same-origin',
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const code: string | undefined = data?.detail?.code;
    if (response.status === 401 && code === 'auth_required') {
      window.dispatchEvent(new CustomEvent(AUTH_REQUIRED_EVENT));
    } else if (response.status === 401 && code === 'profile_required') {
      window.dispatchEvent(new CustomEvent(PROFILE_REQUIRED_EVENT));
    }
    throw new ApiError(errorMessage(data, response.status), response.status, code);
  }

  return response.json();
}

export const api = {
  // Owner Auth
  ownerStatus: () => request<OwnerStatus>('/auth/owner'),
  setupOwner: (passphrase: string, setupCode: string) =>
    request<{ authenticated: boolean }>('/auth/owner/setup', {
      method: 'POST',
      body: JSON.stringify({ passphrase, setupCode }),
    }),
  loginOwner: (passphrase: string) =>
    request<{ authenticated: boolean }>('/auth/owner/login', {
      method: 'POST',
      body: JSON.stringify({ passphrase }),
    }),
  logoutOwner: () =>
    request<{ authenticated: boolean }>('/auth/owner/logout', { method: 'POST' }),
  changePassphrase: (current: string, next: string) =>
    request<{ ok: boolean }>('/auth/owner/passphrase', {
      method: 'POST',
      body: JSON.stringify({ current, new: next }),
    }),

  // Profiles
  listProfiles: () => request<{ profiles: Profile[]; active_id: string | null }>('/profiles'),
  createProfile: (payload: { name: string; color: string; pin?: string }) =>
    request<Profile>('/profiles', { method: 'POST', body: JSON.stringify(payload) }),
  updateProfile: (id: string, payload: { name?: string; color?: string; pin?: string }) =>
    request<Profile>(`/profiles/${id}`, { method: 'PATCH', body: JSON.stringify(payload) }),
  deleteProfile: (id: string) => request<{ ok: boolean }>(`/profiles/${id}`, { method: 'DELETE' }),
  selectProfile: (id: string, pin?: string) =>
    request<Profile>(`/profiles/${id}/select`, { method: 'POST', body: JSON.stringify({ pin }) }),

  // System & Modes
  getModes: () => request<{ modes: CoachingMode[] }>('/modes'),
  getStatus: () => request<any>('/status'),

  // Sessions
  listSessions: (limit = 50) => request<{ sessions: Session[] }>(`/sessions?limit=${limit}`),
  createSession: (title: string, mode = 'pitch_analysis') =>
    request<Session>('/sessions', {
      method: 'POST',
      body: JSON.stringify({ title, mode }),
    }),
  getSession: (id: string) =>
    request<{ session: Session; messages: Message[] }>(`/sessions/${id}`),
  updateMode: (id: string, mode: string) =>
    request<{ ok: boolean }>(`/sessions/${id}/mode`, {
      method: 'PATCH',
      body: JSON.stringify({ mode }),
    }),
  renameSession: (id: string, title: string) =>
    request<{ ok: boolean }>(`/sessions/${id}/rename`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    }),
  deleteSession: (id: string) =>
    request<{ ok: boolean }>(`/sessions/${id}`, { method: 'DELETE' }),
  deleteLastTurn: (id: string) =>
    request<{ deleted: number }>(`/sessions/${id}/turns/last`, { method: 'DELETE' }),
  getProgress: () => request<{ progress: any[] }>('/sessions/progress'),

  // AI Providers & Models
  getAiProviders: () => request<AiProvidersState>('/ai/providers'),
  addAiProvider: (payload: { kind: string; name?: string; baseUrl?: string | null; apiKey?: string | null }) =>
    request<AiProvidersState>('/ai/providers', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  updateAiProvider: (id: string, payload: { name?: string; baseUrl?: string | null; apiKey?: string | null; enabled?: boolean }) =>
    request<AiProvidersState>(`/ai/providers/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  deleteAiProvider: (id: string) =>
    request<AiProvidersState>(`/ai/providers/${id}`, { method: 'DELETE' }),
  fetchModels: (id: string) =>
    request<AiProvidersState>(`/ai/providers/${id}/fetch-models`, { method: 'POST' }),
  testModel: (ref: string, capability = 'chat') =>
    request<{ ok: boolean; detail?: string; error?: string; ms: number }>('/ai/test', {
      method: 'POST',
      body: JSON.stringify({ ref, capability }),
    }),
  getAiPreferences: () => request<AiPreferences>('/ai/preferences'),
  saveAiPreferences: (prefs: Partial<AiPreferences>) =>
    request<AiPreferences>('/ai/preferences', {
      method: 'PUT',
      body: JSON.stringify(prefs),
    }),

  // TTS Read Aloud
  readAloud: (text: string, voice = 'en_US-lessac-medium') =>
    request<{ format: string; data: string }>('/tts/read-aloud', {
      method: 'POST',
      body: JSON.stringify({ text, voice }),
    }),
};
