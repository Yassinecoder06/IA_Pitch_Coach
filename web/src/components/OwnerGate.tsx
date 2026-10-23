import React, { useEffect, useState, type ReactNode } from 'react';
import { Lock, KeyRound, Terminal, ShieldAlert, Sparkles, Eye, EyeOff } from 'lucide-react';
import { api, AUTH_REQUIRED_EVENT } from '../lib/api';

type GateState = 'loading' | 'setup' | 'login' | 'ready' | 'offline';

export function OwnerGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<GateState>('loading');
  const [passphrase, setPassphrase] = useState('');
  const [confirm, setConfirm] = useState('');
  const [setupCode, setSetupCode] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [message, setMessage] = useState('');
  const [working, setWorking] = useState(false);

  async function check() {
    try {
      const status = await api.ownerStatus();
      if (!status.configured) {
        setState('setup');
      } else if (status.authenticated) {
        setState('ready');
      } else {
        setState('login');
      }
    } catch {
      setState('offline');
    }
  }

  useEffect(() => {
    void check();
    const onAuthRequired = () => setState(curr => (curr === 'ready' ? 'login' : curr));
    window.addEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
    return () => window.removeEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setMessage('');

    if (state === 'setup' && passphrase !== confirm) {
      setMessage('Passphrases do not match.');
      return;
    }

    setWorking(true);
    try {
      let res;
      if (state === 'setup') {
        res = await api.setupOwner(passphrase, setupCode);
      } else {
        res = await api.loginOwner(passphrase);
      }
      if (res.token) {
        localStorage.setItem('pitch_coach_token', res.token);
      }
      setPassphrase('');
      setConfirm('');
      setSetupCode('');
      setState('ready');
    } catch (err: any) {
      setMessage(err instanceof Error ? err.message : 'Authentication failed');
    } finally {
      setWorking(false);
    }
  }

  if (state === 'ready') return <>{children}</>;

  if (state === 'loading') {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-[#0b0a10]">
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-purple-500 border-t-transparent" />
          <span className="text-sm text-neutral-400">Opening AI Pitch Coach...</span>
        </div>
      </div>
    );
  }

  if (state === 'offline') {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-[#0b0a10] p-4">
        <div className="glass-panel w-full max-w-md rounded-2xl p-6 text-center">
          <ShieldAlert className="mx-auto mb-3 h-10 w-10 text-amber-400" />
          <h2 className="text-lg font-bold">Backend Not Reachable</h2>
          <p className="mt-1 text-sm text-neutral-400">
            Make sure the FastAPI backend is running on port 8000.
          </p>
          <button
            onClick={() => void check()}
            className="mt-4 rounded-xl bg-purple-600 px-4 py-2 text-sm font-semibold text-white hover:bg-purple-500"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  const setup = state === 'setup';

  return (
    <main className="relative flex min-h-screen w-screen items-center justify-center bg-[#0b0a10] p-4 overflow-hidden">
      {/* Background glow */}
      <div className="pointer-events-none absolute -top-40 left-1/2 -translate-x-1/2 h-[450px] w-[650px] rounded-full bg-purple-600/15 blur-[120px]" />
      <div className="pointer-events-none absolute -bottom-40 right-10 h-[350px] w-[500px] rounded-full bg-indigo-600/10 blur-[100px]" />

      <section className="glass-panel relative z-10 w-full max-w-md rounded-3xl p-8 shadow-2xl shadow-purple-950/20">
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-500 shadow-lg shadow-purple-500/25">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight">AI Pitch Coach</h1>
              <span className="text-xs text-neutral-400">SOTA Local Edition</span>
            </div>
          </div>
          <div className="rounded-full bg-purple-500/10 p-2 text-purple-400">
            <Lock className="h-4 w-4" />
          </div>
        </div>

        <h2 className="text-xl font-bold tracking-tight text-white">
          {setup ? 'Protect Your Pitch Coach' : 'Unlock Pitch Coach'}
        </h2>
        <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed">
          {setup
            ? 'Set a local owner passphrase. Only you with this passphrase can access pitch sessions, transcripts, and configured API keys.'
            : 'Enter your owner passphrase to open your pitch sessions.'}
        </p>

        <form onSubmit={submit} className="mt-6 flex flex-col gap-4">
          {setup && (
            <div>
              <div className="flex items-center justify-between text-xs font-medium text-neutral-300 mb-1.5">
                <span>First-Run Setup Code</span>
                <span className="flex items-center gap-1 text-[11px] text-neutral-500">
                  <Terminal className="h-3 w-3" /> Check server logs
                </span>
              </div>
              <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-neutral-900/60 px-3 py-2 text-sm focus-within:border-purple-500 focus-within:ring-2 focus-within:ring-purple-500/20">
                <KeyRound className="h-4 w-4 text-neutral-500" />
                <input
                  type="text"
                  value={setupCode}
                  onChange={e => setSetupCode(e.target.value)}
                  placeholder="XXXX-XXXX"
                  className="w-full bg-transparent text-sm placeholder:text-neutral-600 focus:outline-none"
                  required
                />
              </div>
            </div>
          )}

          <div>
            <label className="text-xs font-medium text-neutral-300 block mb-1.5">Passphrase</label>
            <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-neutral-900/60 px-3 py-2 text-sm focus-within:border-purple-500 focus-within:ring-2 focus-within:ring-purple-500/20">
              <Lock className="h-4 w-4 text-neutral-500" />
              <input
                type={showPassword ? 'text' : 'password'}
                value={passphrase}
                onChange={e => setPassphrase(e.target.value)}
                minLength={setup ? 8 : 1}
                placeholder={setup ? 'At least 8 characters' : 'Enter passphrase'}
                className="w-full bg-transparent text-sm placeholder:text-neutral-600 focus:outline-none"
                autoFocus
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="text-neutral-500 hover:text-neutral-300"
              >
                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>

          {setup && (
            <div>
              <label className="text-xs font-medium text-neutral-300 block mb-1.5">Confirm Passphrase</label>
              <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-neutral-900/60 px-3 py-2 text-sm focus-within:border-purple-500 focus-within:ring-2 focus-within:ring-purple-500/20">
                <Lock className="h-4 w-4 text-neutral-500" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={confirm}
                  onChange={e => setConfirm(e.target.value)}
                  minLength={8}
                  placeholder="Re-enter passphrase"
                  className="w-full bg-transparent text-sm placeholder:text-neutral-600 focus:outline-none"
                  required
                />
              </div>
            </div>
          )}

          {message && (
            <div className="rounded-xl border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-xs text-rose-400">
              {message}
            </div>
          )}

          <button
            type="submit"
            disabled={working}
            className="mt-2 flex w-full items-center justify-center rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 py-2.5 text-sm font-semibold text-white shadow-lg shadow-purple-600/25 transition-all hover:brightness-110 disabled:opacity-50"
          >
            {working ? 'Verifying...' : setup ? 'Claim & Unlock Coach' : 'Unlock Coach'}
          </button>
        </form>
      </section>
    </main>
  );
}
