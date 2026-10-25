import React, { useEffect, useState } from 'react';
import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import { Check, KeyRound, Monitor, Moon, Sun, Trash2 } from 'lucide-react';
import { AiProvidersPanel } from './AiProvidersPanel';
import { ProfileAvatar, PROFILE_COLORS } from '@/features/profiles/ProfileGate';
import { api } from '@/lib/api';
import { rehearsalsApi, type PrivacySettings } from '@/lib/rehearsals';
import type { Profile } from '@/lib/types';
import { useAuth } from '@/stores/auth';
import { useTheme, type ThemeChoice } from '@/stores/theme';
import { toast } from '@/stores/toast';

const TABS = [
  { to: 'ai', label: 'AI providers' },
  { to: 'profiles', label: 'Profiles' },
  { to: 'security', label: 'Security' },
  { to: 'privacy', label: 'Privacy' },
  { to: 'appearance', label: 'Appearance' },
];

export function SettingsPage() {
  return (
    <div className="h-full overflow-y-auto p-6">
      <h1 className="text-xl font-bold">Settings</h1>
      <div className="mt-4 flex gap-2 border-b border-white/10" role="tablist">
        {TABS.map(t => (
          <NavLink
            key={t.to}
            to={t.to}
            role="tab"
            className={({ isActive }) =>
              `-mb-px border-b-2 px-3 py-2 text-sm ${
                isActive ? 'border-purple-500 text-white' : 'border-transparent text-neutral-400 hover:text-white'
              }`
            }
          >
            {t.label}
          </NavLink>
        ))}
      </div>
      <div className="mt-6">
        <Routes>
          <Route index element={<Navigate to="ai" replace />} />
          <Route path="ai" element={<AiProvidersPanel />} />
          <Route path="profiles" element={<ProfilesSettings />} />
          <Route path="security" element={<SecuritySettings />} />
          <Route path="privacy" element={<PrivacySettingsPanel />} />
          <Route path="appearance" element={<AppearanceSettings />} />
        </Routes>
      </div>
    </div>
  );
}

function ProfilesSettings() {
  const { profile: active, setProfile } = useAuth();
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [newName, setNewName] = useState('');

  const load = () => api.listProfiles().then(r => setProfiles(r.profiles));
  useEffect(() => {
    void load();
  }, []);

  async function save(p: Profile, patch: { name?: string; color?: string; pin?: string }) {
    try {
      const updated = await api.updateProfile(p.id, patch);
      setProfiles(list => list.map(x => (x.id === p.id ? updated : x)));
      if (active?.id === p.id) setProfile(updated);
      toast('Profile saved.');
    } catch (e: any) {
      toast(e.message, 'error');
    }
  }

  async function remove(p: Profile) {
    if (!window.confirm(`Delete "${p.name}" and all of its sessions, decks and rehearsals? This cannot be undone.`)) return;
    try {
      await api.deleteProfile(p.id);
      if (active?.id === p.id) {
        const status = await api.ownerStatus();
        setProfile(status.profile);
      }
      await load();
      toast('Profile deleted.');
    } catch (e: any) {
      toast(e.message, 'error');
    }
  }

  async function add(e: React.FormEvent) {
    e.preventDefault();
    try {
      await api.createProfile({ name: newName.trim(), color: PROFILE_COLORS[profiles.length % PROFILE_COLORS.length] });
      setNewName('');
      await load();
    } catch (err: any) {
      toast(err.message, 'error');
    }
  }

  return (
    <div className="max-w-2xl space-y-3">
      {profiles.map(p => (
        <ProfileRow key={p.id} profile={p} isActive={p.id === active?.id} onSave={save} onDelete={remove} canDelete={profiles.length > 1} />
      ))}
      <form onSubmit={add} className="flex gap-2 pt-2">
        <input
          required
          maxLength={80}
          value={newName}
          onChange={e => setNewName(e.target.value)}
          placeholder="New profile name"
          aria-label="New profile name"
          className="flex-1 rounded-xl border border-white/10 bg-neutral-900 px-3 py-2 text-sm focus:border-purple-500 focus:outline-none"
        />
        <button className="rounded-xl bg-purple-600 px-4 py-2 text-sm font-semibold hover:bg-purple-500">Add profile</button>
      </form>
    </div>
  );
}

function ProfileRow({
  profile,
  isActive,
  canDelete,
  onSave,
  onDelete,
}: {
  profile: Profile;
  isActive: boolean;
  canDelete: boolean;
  onSave: (p: Profile, patch: { name?: string; color?: string; pin?: string }) => void;
  onDelete: (p: Profile) => void;
}) {
  const [name, setName] = useState(profile.name);
  const [pin, setPin] = useState('');

  return (
    <div className="glass-card flex flex-wrap items-center gap-3 rounded-2xl p-4">
      <ProfileAvatar profile={{ name, color: profile.color }} />
      <input
        value={name}
        onChange={e => setName(e.target.value)}
        onBlur={() => name.trim() && name !== profile.name && onSave(profile, { name: name.trim() })}
        aria-label="Profile name"
        className="w-40 rounded-lg border border-white/10 bg-neutral-900 px-2 py-1.5 text-sm focus:border-purple-500 focus:outline-none"
      />
      <div className="flex gap-1" role="radiogroup" aria-label="Color">
        {PROFILE_COLORS.map(c => (
          <button
            key={c}
            role="radio"
            aria-checked={c === profile.color}
            aria-label={c}
            onClick={() => onSave(profile, { color: c })}
            className="flex h-5 w-5 items-center justify-center rounded-full"
            style={{ background: c }}
          >
            {c === profile.color && <Check className="h-3 w-3" />}
          </button>
        ))}
      </div>
      <form
        className="flex items-center gap-1"
        onSubmit={e => {
          e.preventDefault();
          onSave(profile, { pin });
          setPin('');
        }}
      >
        <input
          inputMode="numeric"
          maxLength={4}
          value={pin}
          onChange={e => setPin(e.target.value.replace(/\D/g, ''))}
          placeholder={profile.has_pin ? 'New PIN' : 'Set PIN'}
          aria-label="4-digit PIN"
          className="w-20 rounded-lg border border-white/10 bg-neutral-900 px-2 py-1.5 text-sm focus:border-purple-500 focus:outline-none"
        />
        <button disabled={pin.length !== 4} className="rounded-lg px-2 py-1.5 text-xs text-purple-300 hover:bg-white/5 disabled:opacity-40">
          Save PIN
        </button>
        {profile.has_pin && (
          <button type="button" onClick={() => onSave(profile, { pin: '' })} className="rounded-lg px-2 py-1.5 text-xs text-neutral-400 hover:bg-white/5">
            Remove PIN
          </button>
        )}
      </form>
      {isActive && <span className="rounded-full bg-purple-500/15 px-2 py-0.5 text-[11px] text-purple-300">Active</span>}
      <button
        disabled={!canDelete}
        onClick={() => onDelete(profile)}
        aria-label={`Delete ${profile.name}`}
        className="ml-auto rounded-lg p-2 text-neutral-500 hover:bg-rose-500/10 hover:text-rose-300 disabled:opacity-30"
      >
        <Trash2 className="h-4 w-4" />
      </button>
    </div>
  );
}

function SecuritySettings() {
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [confirm, setConfirm] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (next !== confirm) {
      toast('New passphrases do not match.', 'error');
      return;
    }
    setBusy(true);
    try {
      await api.changePassphrase(current, next);
      setCurrent('');
      setNext('');
      setConfirm('');
      toast('Passphrase changed. Other browsers were signed out.');
    } catch (err: any) {
      toast(err.message, 'error');
    } finally {
      setBusy(false);
    }
  }

  const field = 'w-full rounded-xl border border-white/10 bg-neutral-900 px-3 py-2 text-sm focus:border-purple-500 focus:outline-none';
  return (
    <form onSubmit={submit} className="glass-card max-w-md space-y-3 rounded-2xl p-5">
      <h2 className="flex items-center gap-2 font-semibold">
        <KeyRound className="h-4 w-4 text-purple-400" /> Change owner passphrase
      </h2>
      <input type="password" required value={current} onChange={e => setCurrent(e.target.value)} placeholder="Current passphrase" aria-label="Current passphrase" className={field} />
      <input type="password" required minLength={8} value={next} onChange={e => setNext(e.target.value)} placeholder="New passphrase (8+ characters)" aria-label="New passphrase" className={field} />
      <input type="password" required minLength={8} value={confirm} onChange={e => setConfirm(e.target.value)} placeholder="Confirm new passphrase" aria-label="Confirm new passphrase" className={field} />
      <button disabled={busy} className="rounded-xl bg-purple-600 px-4 py-2 text-sm font-semibold hover:bg-purple-500 disabled:opacity-50">
        {busy ? 'Saving...' : 'Change passphrase'}
      </button>
      <p className="text-xs text-neutral-500">
        Forgot it? Run <code className="text-neutral-300">python -m backend.cli reset-owner</code> on the server, then restart it.
      </p>
    </form>
  );
}

function PrivacySettingsPanel() {
  const [p, setP] = useState<PrivacySettings | null>(null);
  useEffect(() => {
    rehearsalsApi.privacy().then(setP).catch(e => toast(e.message, 'error'));
  }, []);
  if (!p) return null;

  async function save(next: PrivacySettings) {
    const previous = p;
    setP(next); // optimistic: the control reflects the click immediately
    try {
      setP(await rehearsalsApi.savePrivacy(next));
      toast('Privacy settings saved.');
    } catch (e: any) {
      setP(previous);
      toast(e.message, 'error');
    }
  }

  return (
    <div className="glass-card max-w-lg space-y-4 rounded-2xl p-5 text-sm">
      <p className="text-neutral-400">
        Everything stays on this machine. Camera frames are analysed in your browser and never stored or sent; only numbers
        (eye contact, posture) are kept. Audio is transcribed by your local Whisper.
      </p>
      <label className="flex items-center gap-2">
        <input type="checkbox" checked={p.keep_recordings} onChange={e => save({ ...p, keep_recordings: e.target.checked })} />
        Keep audio recordings of rehearsals (for playback and re-analysis)
      </label>
      {p.keep_recordings && (
        <label className="flex items-center gap-2">
          Delete recordings after
          <select
            value={p.recording_retention_days ?? ''}
            onChange={e => save({ ...p, recording_retention_days: e.target.value ? Number(e.target.value) : null })}
            className="rounded-lg border border-white/10 bg-neutral-900 px-2 py-1"
            aria-label="Recording retention"
          >
            <option value="">never</option>
            {[7, 30, 90, 365].map(d => (
              <option key={d} value={d}>
                {d} days
              </option>
            ))}
          </select>
        </label>
      )}
      <p className="text-xs text-neutral-500">Reports and transcripts are kept until you delete them. Turning recordings off deletes audio right after each analysis.</p>
    </div>
  );
}

const THEMES: { id: ThemeChoice; label: string; icon: typeof Sun }[] = [
  { id: 'dark', label: 'Dark', icon: Moon },
  { id: 'light', label: 'Light', icon: Sun },
  { id: 'system', label: 'Match system', icon: Monitor },
];

function AppearanceSettings() {
  const { choice, setChoice } = useTheme();
  return (
    <div className="glass-card max-w-lg space-y-3 rounded-2xl p-5 text-sm">
      <h2 className="font-semibold">Theme</h2>
      <div className="grid grid-cols-3 gap-2" role="radiogroup" aria-label="Theme">
        {THEMES.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            role="radio"
            aria-checked={choice === id}
            onClick={() => setChoice(id)}
            className={`flex flex-col items-center gap-2 rounded-xl border px-3 py-4 ${
              choice === id ? 'border-purple-500 bg-purple-500/10 text-purple-300' : 'border-white/10 text-neutral-400 hover:bg-white/5'
            }`}
          >
            <Icon className="h-5 w-5" />
            {label}
          </button>
        ))}
      </div>
      <p className="text-xs text-neutral-500">Saved in this browser. Printed reports always use a light page.</p>
    </div>
  );
}
