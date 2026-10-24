import React, { useState } from 'react';
import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import {
  Activity,
  Gamepad2,
  History,
  Layers,
  Lock,
  MessagesSquare,
  Presentation,
  Settings,
  Sparkles,
} from 'lucide-react';
import { ProfileAvatar } from '@/features/profiles/ProfileGate';
import { useAuth } from '@/stores/auth';
import { useToast } from '@/stores/toast';

const NAV = [
  { to: '/coach', label: 'Coach', icon: MessagesSquare },
  { to: '/practice', label: 'Practice', icon: Activity },
  { to: '/rehearse', label: 'Rehearse', icon: Presentation },
  { to: '/decks', label: 'Decks', icon: Layers },
  { to: '/history', label: 'History', icon: History },
  { to: '/arcade', label: 'Arcade', icon: Gamepad2 },
  { to: '/settings', label: 'Settings', icon: Settings },
];

export function Shell() {
  const { profile, lock, setProfile } = useAuth();
  const toast = useToast();
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0b0a10] text-white">
      <nav className="flex w-16 shrink-0 flex-col items-center gap-1 border-r border-white/5 bg-neutral-950/80 py-4" aria-label="Main">
        <div className="mb-4 flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-500">
          <Sparkles className="h-4 w-4" />
        </div>
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            title={label}
            className={({ isActive }) =>
              `flex w-14 flex-col items-center gap-0.5 rounded-xl py-2 text-[10px] transition-colors ${
                isActive ? 'bg-purple-600/20 text-purple-300' : 'text-neutral-500 hover:bg-white/5 hover:text-neutral-200'
              }`
            }
          >
            <Icon className="h-5 w-5" />
            {label}
          </NavLink>
        ))}

        <div className="relative mt-auto flex flex-col items-center gap-2">
          {profile && (
            <button
              onClick={() => setMenuOpen(o => !o)}
              title={`Profile: ${profile.name}`}
              aria-label={`Profile ${profile.name}. Switch profile`}
              className="rounded-full focus:outline-none focus-visible:ring-2 focus-visible:ring-purple-500"
            >
              <ProfileAvatar profile={profile} size={34} />
            </button>
          )}
          {menuOpen && profile && (
            <div className="absolute bottom-0 left-14 z-50 w-48 rounded-xl border border-white/10 bg-neutral-900 p-1 text-sm shadow-2xl">
              <div className="px-3 py-2 text-xs text-neutral-400">Signed in as {profile.name}</div>
              <button
                className="w-full rounded-lg px-3 py-2 text-left hover:bg-white/5"
                onClick={() => {
                  setMenuOpen(false);
                  setProfile(null); // ProfileGate shows the picker
                }}
              >
                Switch profile
              </button>
              <button
                className="w-full rounded-lg px-3 py-2 text-left hover:bg-white/5"
                onClick={() => {
                  setMenuOpen(false);
                  navigate('/settings/profiles');
                }}
              >
                Manage profiles
              </button>
            </div>
          )}
          <button
            onClick={() => void lock()}
            title="Lock Pitch Coach"
            aria-label="Lock Pitch Coach"
            className="rounded-xl p-2 text-neutral-500 hover:bg-white/5 hover:text-rose-300"
          >
            <Lock className="h-4 w-4" />
          </button>
        </div>
      </nav>

      <main className="min-w-0 flex-1 overflow-hidden">
        <Outlet />
      </main>

      {toast.message && (
        <div
          role="status"
          className={`fixed bottom-6 right-6 z-50 flex items-center gap-2 rounded-2xl border px-4 py-2.5 text-xs shadow-2xl ${
            toast.tone === 'error' ? 'border-rose-500/30 bg-rose-950 text-rose-100' : 'border-purple-500/30 bg-neutral-900 text-white'
          }`}
        >
          <span className={`h-2 w-2 rounded-full ${toast.tone === 'error' ? 'bg-rose-400' : 'bg-purple-400'}`} />
          {toast.message}
        </div>
      )}
    </div>
  );
}

export function ComingSoon({ title }: { title: string }) {
  return (
    <div className="flex h-full items-center justify-center text-neutral-500">
      <p>{title} is not available yet.</p>
    </div>
  );
}
