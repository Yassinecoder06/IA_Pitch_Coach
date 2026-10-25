import { create } from 'zustand';

interface ToastState {
  message: string | null;
  tone: 'info' | 'error';
  show: (message: string, tone?: 'info' | 'error') => void;
}

let timer: ReturnType<typeof setTimeout> | undefined;

export const useToast = create<ToastState>(set => ({
  message: null,
  tone: 'info',
  show: (message, tone = 'info') => {
    clearTimeout(timer);
    set({ message, tone });
    timer = setTimeout(() => set({ message: null }), 3500);
  },
}));

export const toast = (message: string, tone: 'info' | 'error' = 'info') => useToast.getState().show(message, tone);
