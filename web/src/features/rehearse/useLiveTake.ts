import { useCallback, useEffect, useRef, useState } from 'react';
import { rehearsalSocket } from '@/lib/ws';
import { SensorSession, type SensorSummary } from '@/lib/sensors/session';
import { useSensors, type LiveWord } from '@/stores/sensors';
import { PCM_SAMPLE_RATE } from '@/audio/pcmRecorder';

export type TakePhase = 'idle' | 'starting' | 'live' | 'finishing' | 'analysing' | 'done' | 'error';

export interface TakeResult {
  text: string;
  words: LiveWord[];
  duration: number;
  sensors: SensorSummary;
  rehearsalId: string | null;
}

interface StartOptions {
  camera: boolean;
  mic: boolean;
  video: HTMLVideoElement | null;
  rehearsalId?: string | null;
}

/** Runs one recorded take: sensors in the browser, live captions and analysis on the server. */
export function useLiveTake() {
  const [phase, setPhase] = useState<TakePhase>('idle');
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TakeResult | null>(null);
  const session = useRef<SensorSession | null>(null);
  const finalWaiter = useRef<((v: { text: string; words: LiveWord[]; duration: number }) => void) | null>(null);
  const doneWaiter = useRef<(() => void) | null>(null);
  const { apply, addWords, reset } = useSensors.getState();

  useEffect(() => {
    const off = rehearsalSocket.subscribe(msg => {
      switch (msg.type) {
        case 'words':
          addWords(msg.words, msg.fillers);
          break;
        case 'transcript_final':
          finalWaiter.current?.({ text: msg.text, words: msg.words, duration: msg.duration });
          break;
        case 'analysis_progress':
          setProgress(msg.step);
          break;
        case 'analysis_done':
          doneWaiter.current?.();
          break;
        case 'error':
          setError(msg.message);
          setPhase('error');
          break;
      }
    });
    return () => {
      off();
      void session.current?.stop();
    };
  }, [addWords]);

  const start = useCallback(
    async ({ camera, mic, video, rehearsalId = null }: StartOptions) => {
      setError(null);
      setResult(null);
      setProgress(null);
      reset();
      setPhase('starting');
      try {
        await rehearsalSocket.open();
        const s = new SensorSession({
          camera,
          mic,
          video,
          onFrame: apply,
          onPcm: frame => rehearsalSocket.sendBinary(frame),
        });
        rehearsalSocket.send({ type: 'start', sample_rate: PCM_SAMPLE_RATE, rehearsal_id: rehearsalId });
        await s.start();
        session.current = s;
        setPhase('live');
        return s;
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setPhase('error');
        return null;
      }
    },
    [apply, reset],
  );

  const slide = useCallback((index: number) => {
    const s = session.current;
    if (s) rehearsalSocket.send({ type: 'slide', index, t: s.elapsed });
  }, []);

  const stop = useCallback(async (rehearsalId: string | null = null): Promise<TakeResult | null> => {
    const s = session.current;
    if (!s) return null;
    session.current = null;
    setPhase('finishing');
    const sensors = await s.stop();
    const finalPromise = new Promise<{ text: string; words: LiveWord[]; duration: number }>(r => (finalWaiter.current = r));
    const donePromise = rehearsalId ? new Promise<void>(r => (doneWaiter.current = r)) : Promise.resolve();
    rehearsalSocket.send({ type: 'stop', sensors });
    const final = await finalPromise;
    if (rehearsalId) {
      setPhase('analysing');
      await donePromise;
    }
    const res = { ...final, sensors, rehearsalId };
    setResult(res);
    setPhase('done');
    return res;
  }, []);

  const calibrate = useCallback(async () => session.current?.calibrate() ?? null, []);

  return { phase, progress, error, result, start, stop, slide, calibrate, session };
}
