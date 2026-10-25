import { create } from 'zustand';

export type ThemeChoice = 'dark' | 'light' | 'system';

/** Shared with the inline script in index.html, which applies the theme before first paint. */
const STORAGE_KEY = 'pitch-coach-theme';
const media = typeof window !== 'undefined' ? window.matchMedia('(prefers-color-scheme: light)') : null;

function readChoice(): ThemeChoice {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === 'light' || v === 'dark' || v === 'system' ? v : 'dark';
  } catch {
    return 'dark';
  }
}

const resolve = (choice: ThemeChoice): 'dark' | 'light' => (choice === 'system' ? (media?.matches ? 'light' : 'dark') : choice);

function apply(choice: ThemeChoice) {
  document.documentElement.dataset.theme = resolve(choice);
}

interface ThemeState {
  choice: ThemeChoice;
  setChoice: (choice: ThemeChoice) => void;
  toggle: () => void;
}

export const useTheme = create<ThemeState>((set, get) => ({
  choice: readChoice(),
  setChoice: choice => {
    try {
      localStorage.setItem(STORAGE_KEY, choice);
    } catch {
      // Private mode or blocked storage: the choice still applies for this visit.
    }
    apply(choice);
    set({ choice });
  },
  toggle: () => get().setChoice(resolve(get().choice) === 'dark' ? 'light' : 'dark'),
}));

/** The theme actually shown ('system' resolved). */
export const useResolvedTheme = () => resolve(useTheme(s => s.choice));

apply(useTheme.getState().choice);
media?.addEventListener('change', () => {
  if (useTheme.getState().choice === 'system') apply('system');
});
