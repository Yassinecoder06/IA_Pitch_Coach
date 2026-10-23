import { create } from 'zustand';
import type { Profile } from '@/lib/types';
import { api } from '@/lib/api';
import { coachSocket } from '@/lib/ws';
import { queryClient } from '@/lib/queryClient';

interface AuthState {
  profile: Profile | null;
  setProfile: (p: Profile | null) => void;
  /** Switches the active profile: new cookie, fresh caches, and a socket bound to the new profile. */
  switchProfile: (id: string, pin?: string) => Promise<Profile>;
  lock: () => Promise<void>;
}

export const useAuth = create<AuthState>(set => ({
  profile: null,
  setProfile: profile => set({ profile }),
  switchProfile: async (id, pin) => {
    const profile = await api.selectProfile(id, pin);
    coachSocket.disconnect();
    queryClient.clear();
    set({ profile });
    return profile;
  },
  lock: async () => {
    await api.logoutOwner().catch(() => undefined);
    coachSocket.disconnect();
    queryClient.clear();
    window.location.assign('/');
  },
}));
