import React, { useEffect, useState } from 'react';
import {
  Server,
  Plus,
  RefreshCw,
  KeyRound,
  Trash2,
  Check,
  X,
  Zap,
  ArrowUp,
  ArrowDown,
  Loader2,
  Sparkles,
  ExternalLink,
} from 'lucide-react';
import { api } from '../lib/api';
import type { AiPreferences, AiProvidersState } from '../lib/types';

export function AiProvidersPanel({
  onClose,
  onToast,
}: {
  onClose: () => void;
  onToast: (msg: string) => void;
}) {
  const [state, setState] = useState<AiProvidersState | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [adding, setAdding] = useState(false);
  const [form, setForm] = useState({ kind: 'openai', name: '', baseUrl: '', apiKey: '' });
  const [prefs, setPrefs] = useState<AiPreferences | null>(null);
  const [testResults, setTestResults] = useState<Record<string, { ok: boolean; text: string } | 'running'>>({});

  const load = (next: AiProvidersState) => {
    setState(next);
    setPrefs(next.preferences);
  };

  useEffect(() => {
    api.getAiProviders()
      .then(load)
      .catch(err => setError(err instanceof Error ? err.message : 'Could not load AI providers.'));
  }, []);

  async function addProvider() {
    setBusy('add');
    setError('');
    try {
      const next = await api.addAiProvider({
        kind: form.kind,
        name: form.name.trim() || undefined,
        baseUrl: form.baseUrl.trim() || null,
        apiKey: form.apiKey.trim() || null,
      });
      load(next);
      setForm({ kind: 'openai', name: '', baseUrl: '', apiKey: '' });
      setAdding(false);
      onToast('Provider added and models fetched.');
    } catch (err: any) {
      setError(err instanceof Error ? err.message : 'Failed to add provider.');
    } finally {
      setBusy(null);
    }
  }

  async function removeProvider(id: string) {
    setBusy(id);
    try {
      const next = await api.deleteAiProvider(id);
      load(next);
      onToast('Provider removed.');
    } catch (err: any) {
      setError(err instanceof Error ? err.message : 'Failed to delete provider.');
    } finally {
      setBusy(null);
    }
  }

  async function refreshModels(id: string) {
    setBusy(id);
    try {
      const next = await api.fetchModels(id);
      load(next);
      onToast('Models refreshed.');
    } catch (err: any) {
      setError(err instanceof Error ? err.message : 'Failed to fetch models.');
    } finally {
      setBusy(null);
    }
  }

  async function runTest(ref: string) {
    setTestResults(prev => ({ ...prev, [ref]: 'running' }));
    try {
      const res = await api.testModel(ref);
      setTestResults(prev => ({
        ...prev,
        [ref]: {
          ok: res.ok,
          text: res.ok ? `${res.detail} (${(res.ms / 1000).toFixed(1)}s)` : res.error || 'Failed',
        },
      }));
    } catch (err: any) {
      setTestResults(prev => ({
        ...prev,
        [ref]: { ok: false, text: err instanceof Error ? err.message : 'Test failed' },
      }));
    }
  }

  async function savePreferences() {
    if (!prefs) return;
    setBusy('prefs');
    try {
      const next = await api.saveAiPreferences(prefs);
      setPrefs(next);
      onToast('AI choices saved.');
    } catch (err: any) {
      setError(err instanceof Error ? err.message : 'Failed to save choices.');
    } finally {
      setBusy(null);
    }
  }

  if (!state || !prefs) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-md p-4">
        <div className="glass-panel w-full max-w-2xl rounded-2xl p-8 text-center">
          <Loader2 className="mx-auto h-8 w-8 animate-spin text-purple-400" />
          <p className="mt-2 text-sm text-neutral-400">Loading AI configuration...</p>
        </div>
      </div>
    );
  }

  const primaryModel = prefs.chat_models[0] || 'builtin:qwen3:0.6b';
  const fallbacks = prefs.fallback_models;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-md p-4 overflow-y-auto">
      <div className="glass-panel my-8 w-full max-w-3xl rounded-3xl p-6 sm:p-8 shadow-2xl shadow-purple-950/40">
        <div className="flex items-center justify-between border-b border-white/10 pb-5">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-600/20 text-purple-400">
              <Zap className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">AI Providers & Keys</h2>
              <p className="text-xs text-neutral-400">
                Connect OpenAI, Claude, Gemini, Groq, DeepSeek, or local Ollama.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-xl border border-white/10 p-2 text-neutral-400 hover:bg-white/5 hover:text-white"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {error && (
          <div className="mt-4 flex items-center justify-between rounded-xl border border-rose-500/20 bg-rose-500/10 px-4 py-2.5 text-xs text-rose-300">
            <span>{error}</span>
            <button onClick={() => setError('')}>
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}

        {/* Configured Providers */}
        <section className="mt-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
              Your Providers ({state.providers.length})
            </h3>
            {!adding && (
              <button
                onClick={() => setAdding(true)}
                className="flex items-center gap-1.5 rounded-xl bg-purple-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-purple-500"
              >
                <Plus className="h-3.5 w-3.5" /> Add Provider
              </button>
            )}
          </div>

          {adding && (
            <div className="glass-card mb-4 rounded-2xl p-5 border border-purple-500/30">
              <h4 className="text-sm font-semibold text-white mb-3">Add AI Provider</h4>
              <div className="grid gap-3 sm:grid-cols-2">
                <div>
                  <label className="text-xs text-neutral-300 block mb-1">Provider Kind</label>
                  <select
                    value={form.kind}
                    onChange={e => setForm({ ...form, kind: e.target.value })}
                    className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3 py-2 text-xs text-white focus:outline-none focus:border-purple-500"
                  >
                    {state.kinds.map(k => (
                      <option key={k.kind} value={k.kind}>
                        {k.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-neutral-300 block mb-1">Display Name (Optional)</label>
                  <input
                    type="text"
                    placeholder="e.g. My Groq or Work OpenAI"
                    value={form.name}
                    onChange={e => setForm({ ...form, name: e.target.value })}
                    className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3 py-2 text-xs text-white placeholder:text-neutral-600 focus:outline-none focus:border-purple-500"
                  />
                </div>
                <div className="sm:col-span-2">
                  <label className="text-xs text-neutral-300 block mb-1">
                    Base URL {form.kind.endsWith('_compatible') ? '(Required)' : '(Optional)'}
                  </label>
                  <input
                    type="text"
                    placeholder="e.g. https://api.groq.com/openai/v1 or http://localhost:11434"
                    value={form.baseUrl}
                    onChange={e => setForm({ ...form, baseUrl: e.target.value })}
                    className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3 py-2 text-xs text-white placeholder:text-neutral-600 focus:outline-none focus:border-purple-500"
                  />
                </div>
                <div className="sm:col-span-2">
                  <label className="text-xs text-neutral-300 block mb-1">API Key</label>
                  <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-neutral-900 px-3 py-2 text-xs">
                    <KeyRound className="h-3.5 w-3.5 text-neutral-500" />
                    <input
                      type="password"
                      placeholder="sk-... or gsk_..."
                      value={form.apiKey}
                      onChange={e => setForm({ ...form, apiKey: e.target.value })}
                      className="w-full bg-transparent text-white placeholder:text-neutral-600 focus:outline-none"
                    />
                  </div>
                </div>
              </div>

              <div className="mt-4 flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setAdding(false)}
                  className="rounded-xl border border-white/10 px-3 py-1.5 text-xs text-neutral-300 hover:bg-white/5"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={addProvider}
                  disabled={busy === 'add'}
                  className="rounded-xl bg-purple-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-purple-500 disabled:opacity-50"
                >
                  {busy === 'add' ? 'Connecting...' : 'Add & Fetch Models'}
                </button>
              </div>
            </div>
          )}

          <div className="grid gap-2.5 sm:grid-cols-2">
            {state.providers.map(p => (
              <div key={p.id} className="glass-card flex flex-col justify-between rounded-2xl p-4">
                <div className="flex items-start justify-between">
                  <div>
                    <h4 className="text-sm font-semibold text-white">{p.name}</h4>
                    <span className="text-[11px] text-neutral-400 capitalize">{p.kind.replace('_', ' ')}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => refreshModels(p.id)}
                      disabled={busy === p.id}
                      title="Fetch models"
                      className="rounded-lg p-1.5 text-neutral-400 hover:bg-white/5 hover:text-white"
                    >
                      <RefreshCw className={`h-3.5 w-3.5 ${busy === p.id ? 'animate-spin' : ''}`} />
                    </button>
                    <button
                      onClick={() => removeProvider(p.id)}
                      title="Delete provider"
                      className="rounded-lg p-1.5 text-neutral-400 hover:bg-rose-500/10 hover:text-rose-400"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>

                <div className="mt-3 flex items-center justify-between text-[11px] text-neutral-400">
                  <span>{p.models.length} models available</span>
                  {p.has_key && <span className="font-mono text-neutral-500">{p.key_hint}</span>}
                </div>
              </div>
            ))}

            {state.providers.length === 0 && !adding && (
              <div className="col-span-2 rounded-2xl border border-dashed border-white/10 p-6 text-center text-xs text-neutral-400">
                No custom providers added yet. Using local Ollama default. Click "Add Provider" to add your API keys.
              </div>
            )}
          </div>
        </section>

        {/* Model Preferences & Routing */}
        <section className="mt-8 border-t border-white/10 pt-6">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 mb-4">
            Model Assignment & Fallbacks
          </h3>

          <div className="space-y-4">
            {/* Primary Coaching Model */}
            <div className="glass-card rounded-2xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <strong className="text-sm text-white block">Primary Pitch Coach Model</strong>
                <span className="text-xs text-neutral-400">
                  The model used for real-time speech analysis, structured feedback, and scoring.
                </span>
              </div>
              <div className="flex items-center gap-2">
                <select
                  value={primaryModel}
                  onChange={e => {
                    const newChat = [e.target.value, ...prefs.chat_models.filter(m => m !== e.target.value)];
                    setPrefs({ ...prefs, chat_models: newChat });
                  }}
                  className="rounded-xl border border-white/10 bg-neutral-900 px-3 py-2 text-xs text-white focus:outline-none focus:border-purple-500"
                >
                  {state.catalog.map(m => (
                    <option key={m.ref} value={m.ref}>
                      {m.providerName} · {m.label}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => runTest(primaryModel)}
                  className="rounded-xl border border-white/10 bg-white/5 px-2.5 py-1.5 text-xs text-neutral-300 hover:bg-white/10"
                >
                  Test
                </button>
              </div>
            </div>

            {testResults[primaryModel] && (
              <div className="text-xs px-2 flex items-center gap-1.5 text-neutral-300">
                {testResults[primaryModel] === 'running' ? (
                  <span className="flex items-center gap-1 text-purple-400">
                    <Loader2 className="h-3 w-3 animate-spin" /> Testing model latency...
                  </span>
                ) : testResults[primaryModel].ok ? (
                  <span className="flex items-center gap-1 text-emerald-400">
                    <Check className="h-3.5 w-3.5" /> {testResults[primaryModel].text}
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-rose-400">
                    <X className="h-3.5 w-3.5" /> {testResults[primaryModel].text}
                  </span>
                )}
              </div>
            )}

            {/* Fallback Cascades */}
            <div className="glass-card rounded-2xl p-4">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <strong className="text-sm text-white block">Fallback Cascade</strong>
                  <span className="text-xs text-neutral-400">
                    Automatically used in order if the primary model is busy, rate-limited, or fails.
                  </span>
                </div>
              </div>

              <div className="space-y-1.5 mt-3">
                {fallbacks.map((ref, idx) => {
                  const m = state.catalog.find(c => c.ref === ref);
                  return (
                    <div
                      key={ref}
                      className="flex items-center justify-between rounded-xl bg-neutral-900/60 px-3 py-2 text-xs"
                    >
                      <span className="text-neutral-300">{m ? `${m.providerName} · ${m.label}` : ref}</span>
                      <div className="flex items-center gap-1">
                        <button
                          disabled={idx === 0}
                          onClick={() => {
                            const arr = [...fallbacks];
                            [arr[idx - 1], arr[idx]] = [arr[idx], arr[idx - 1]];
                            setPrefs({ ...prefs, fallback_models: arr });
                          }}
                          className="p-1 text-neutral-400 hover:text-white disabled:opacity-30"
                        >
                          <ArrowUp className="h-3.5 w-3.5" />
                        </button>
                        <button
                          disabled={idx === fallbacks.length - 1}
                          onClick={() => {
                            const arr = [...fallbacks];
                            [arr[idx + 1], arr[idx]] = [arr[idx], arr[idx + 1]];
                            setPrefs({ ...prefs, fallback_models: arr });
                          }}
                          className="p-1 text-neutral-400 hover:text-white disabled:opacity-30"
                        >
                          <ArrowDown className="h-3.5 w-3.5" />
                        </button>
                        <button
                          onClick={() => {
                            setPrefs({
                              ...prefs,
                              fallback_models: fallbacks.filter(f => f !== ref),
                            });
                          }}
                          className="p-1 text-neutral-400 hover:text-rose-400"
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>
                  );
                })}

                <select
                  value=""
                  onChange={e => {
                    if (e.target.value && !fallbacks.includes(e.target.value)) {
                      setPrefs({ ...prefs, fallback_models: [...fallbacks, e.target.value] });
                    }
                  }}
                  className="w-full rounded-xl border border-dashed border-white/10 bg-transparent px-3 py-2 text-xs text-neutral-400 focus:outline-none"
                >
                  <option value="">+ Add a fallback model...</option>
                  {state.catalog.map(m => (
                    <option key={m.ref} value={m.ref}>
                      {m.providerName} · {m.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          <div className="mt-6 flex items-center justify-end gap-3">
            <button
              type="button"
              onClick={onClose}
              className="rounded-xl border border-white/10 px-4 py-2 text-xs font-semibold text-neutral-300 hover:bg-white/5"
            >
              Close
            </button>
            <button
              type="button"
              onClick={savePreferences}
              disabled={busy === 'prefs'}
              className="rounded-xl bg-purple-600 px-5 py-2 text-xs font-semibold text-white shadow-lg shadow-purple-600/20 hover:bg-purple-500 disabled:opacity-50"
            >
              {busy === 'prefs' ? 'Saving...' : 'Save AI Choices'}
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
