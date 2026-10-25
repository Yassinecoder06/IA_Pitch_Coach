import React, { useEffect, useState, type ReactNode } from 'react';
import { Lock, Plus, UserRound } from 'lucide-react';
import { api, PROFILE_REQUIRED_EVENT } from '@/lib/api';
import type { Profile } from '@/lib/types';
import { useAuth } from '@/stores/auth';

export const PROFILE_COLORS = ['#a855f7', '#6366f1', '#0ea5e9', '#10b981', '#f59e0b', '#ef4444', '#ec4899'];

export function ProfileAvatar({ profile, size = 40 }: { profile: Pick<Profile, 'name' | 'color'>; size?: number }) {
  return (
    <span
      className="inline-flex shrink-0 items-center justify-center rounded-full font-semibold text-white"
      style={{ background: profile.color, width: size, height: size, fontSize: size * 0.4 }}
      aria-hidden
    >
      {profile.name.trim().charAt(0).toUpperCase() || '?'}
    </span>
  );
}

/** Shows the profile picker until a profile is active, then renders the app. */
export function ProfileGate({ children }: { children: ReactNode }) {
  const { profile, setProfile } = useAuth();
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    api
      .ownerStatus()
      .then(s => setProfile(s.profile))
      .finally(() => setChecked(true));
    const onRequired = () => setProfile(null);
    window.addEventListener(PROFILE_REQUIRED_EVENT, onRequired);
    return () => window.removeEventListener(PROFILE_REQUIRED_EVENT, onRequired);
  }, [setProfile]);

  if (!checked) return null;
  if (profile) return <>{children}</>;
  return <ProfilePicker />;
}

export function ProfilePicker() {
  const { switchProfile, lock } = useAuth();
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [pinFor, setPinFor] = useState<Profile | null>(null);
  const [pin, setPin] = useState('');
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState('');
  const [error, setError] = useState('');

  const load = () => api.listProfiles().then(r => setProfiles(r.profiles)).catch(e => setError(e.message));
  useEffect(() => {
    void load();
  }, []);

  async function choose(p: Profile, withPin?: string) {
    setError('');
    if (p.has_pin && withPin === undefined) {
      setPinFor(p);
      setPin('');
      return;
    }
    try {
      await switchProfile(p.id, withPin);
    } catch (e: any) {
      setError(e.message);
    }
  }

  async function create(e: React.FormEvent) {
    e.preventDefault();
    try {
      const p = await api.createProfile({ name: name.trim(), color: PROFILE_COLORS[profiles.length % PROFILE_COLORS.length] });
      setCreating(false);
      setName('');
      await switchProfile(p.id);
    } catch (err: any) {
      setError(err.message);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-app p-6">
      <section className="w-full max-w-2xl text-center">
        <h1 className="text-2xl font-bold text-white">Who is practicing?</h1>
        <p className="mt-1 text-sm text-neutral-400">Each profile keeps its own sessions, decks and rehearsal history.</p>

        <div className="mt-8 flex flex-wrap justify-center gap-6">
          {profiles.map(p => (
            <button
              key={p.id}
              onClick={() => void choose(p)}
              className="group flex w-28 flex-col items-center gap-2 rounded-2xl p-3 hover:bg-white/5 focus:outline-none focus-visible:ring-2 focus-visible:ring-purple-500"
            >
              <ProfileAvatar profile={p} size={72} />
              <span className="flex items-center gap-1 text-sm text-neutral-200">
                {p.name}
                {p.has_pin && <Lock className="h-3 w-3 text-neutral-500" aria-label="PIN protected" />}
              </span>
            </button>
          ))}
          <button
            onClick={() => setCreating(true)}
            className="flex w-28 flex-col items-center gap-2 rounded-2xl p-3 text-neutral-400 hover:bg-white/5 hover:text-white"
          >
            <span className="flex h-[72px] w-[72px] items-center justify-center rounded-full border-2 border-dashed border-white/20">
              <Plus className="h-6 w-6" />
            </span>
            <span className="text-sm">Add profile</span>
          </button>
        </div>

        {pinFor && (
          <form
            onSubmit={e => {
              e.preventDefault();
              void choose(pinFor, pin);
            }}
            className="mx-auto mt-8 flex max-w-xs items-center gap-2"
          >
            <input
              autoFocus
              inputMode="numeric"
              maxLength={4}
              value={pin}
              onChange={e => setPin(e.target.value.replace(/\D/g, ''))}
              placeholder={`PIN for ${pinFor.name}`}
              aria-label={`PIN for ${pinFor.name}`}
              className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3 py-2 text-center text-lg tracking-[0.5em] text-white focus:border-purple-500 focus:outline-none"
            />
            <button className="rounded-xl bg-purple-600 px-4 py-2 text-sm font-semibold text-white hover:bg-purple-500">Open</button>
          </form>
        )}

        {creating && (
          <form onSubmit={create} className="mx-auto mt-8 flex max-w-xs items-center gap-2">
            <UserRound className="h-5 w-5 text-neutral-500" />
            <input
              autoFocus
              required
              maxLength={80}
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="Profile name"
              aria-label="Profile name"
              className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3 py-2 text-sm text-white focus:border-purple-500 focus:outline-none"
            />
            <button className="rounded-xl bg-purple-600 px-4 py-2 text-sm font-semibold text-white hover:bg-purple-500">Create</button>
          </form>
        )}

        {error && <p className="mt-4 text-sm text-rose-400" role="alert">{error}</p>}

        <button
          onClick={() => void lock()}
          className="mt-10 inline-flex items-center gap-1.5 text-xs text-neutral-500 hover:text-neutral-200"
        >
          <Lock className="h-3.5 w-3.5" /> Lock Pitch Coach
        </button>
      </section>
    </main>
  );
}
