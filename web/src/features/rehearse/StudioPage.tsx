import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ChevronLeft, ChevronRight, Crosshair, Loader2, Play, Square, StickyNote } from 'lucide-react';
import { decksApi, type Deck } from '@/lib/decks';
import { rehearsalsApi } from '@/lib/rehearsals';
import { SlideView } from '@/features/decks/SlideView';
import { CameraView } from './components/CameraView';
import { Hud } from './components/Hud';
import { LiveCaptions } from './components/LiveCaptions';
import { useLiveTake } from './useLiveTake';
import { toast } from '@/stores/toast';

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

export function StudioPage() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const decks = useQuery({ queryKey: ['decks'], queryFn: decksApi.list });
  const [deckId, setDeckId] = useState<string>(params.get('deck') ?? '');
  const deck = useQuery({ queryKey: ['deck', deckId], queryFn: () => decksApi.get(deckId), enabled: !!deckId });
  const [minutes, setMinutes] = useState<number>(3);
  const [camera, setCamera] = useState(true);
  const [title, setTitle] = useState('');
  const [rehearsalId, setRehearsalId] = useState<string | null>(null);
  const take = useLiveTake();
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Default target length: the deck's slide budgets.
  useEffect(() => {
    const total = deck.data?.slides.reduce((s, x) => s + (x.time_budget_sec ?? 0), 0) ?? 0;
    if (total > 0) setMinutes(Math.round((total / 60) * 2) / 2);
  }, [deck.data]);

  async function start() {
    try {
      const r = await rehearsalsApi.create({
        title: title.trim() || `${deck.data?.title ?? 'Free talk'} · ${new Date().toLocaleString()}`,
        deck_id: deckId || null,
        target_seconds: Math.round(minutes * 60),
      });
      setRehearsalId(r.id);
      await take.start({ camera, mic: true, video: videoRef.current, rehearsalId: r.id });
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Could not start', 'error');
    }
  }

  async function stop() {
    if (!rehearsalId) return;
    const res = await take.stop(rehearsalId);
    if (res?.rehearsalId) navigate(`/report/${res.rehearsalId}`);
  }

  const live = take.phase === 'live';
  const busy = take.phase === 'finishing' || take.phase === 'analysing';

  return (
    <div className="h-full overflow-hidden">
      {/* The camera element must exist before start() so the stream can attach to it. */}
      <div className={live || busy ? 'hidden' : 'h-full overflow-y-auto p-6'}>
        <Setup
          decks={decks.data?.decks ?? []}
          deckId={deckId}
          setDeckId={setDeckId}
          deck={deck.data}
          minutes={minutes}
          setMinutes={setMinutes}
          camera={camera}
          setCamera={setCamera}
          title={title}
          setTitle={setTitle}
          starting={take.phase === 'starting'}
          onStart={start}
          error={take.error}
        />
      </div>
      <Presenter
        visible={live || busy}
        deck={deck.data}
        camera={camera}
        videoRef={videoRef}
        targetSeconds={minutes * 60}
        take={take}
        onStop={stop}
      />
      {busy && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70" role="status">
          <div className="rounded-2xl bg-neutral-900 p-6 text-center">
            <Loader2 className="mx-auto h-8 w-8 animate-spin text-purple-400" />
            <p className="mt-3 font-semibold">{take.phase === 'finishing' ? 'Finishing the transcript...' : 'Analysing your rehearsal...'}</p>
            {take.progress && <p className="mt-1 text-sm text-neutral-400">{take.progress}</p>}
          </div>
        </div>
      )}
    </div>
  );
}

function Setup(props: {
  decks: { id: string; title: string; slide_count: number }[];
  deckId: string;
  setDeckId: (v: string) => void;
  deck?: Deck;
  minutes: number;
  setMinutes: (v: number) => void;
  camera: boolean;
  setCamera: (v: boolean) => void;
  title: string;
  setTitle: (v: string) => void;
  starting: boolean;
  onStart: () => void;
  error: string | null;
}) {
  const field = 'w-full rounded-xl border border-white/10 bg-neutral-900 px-3 py-2 text-sm focus:border-purple-500 focus:outline-none';
  return (
    <div className="mx-auto max-w-2xl space-y-4">
      <h1 className="text-xl font-bold">Rehearsal Studio</h1>
      <p className="text-sm text-neutral-400">
        Present your deck as if it were the real thing. Use the arrow keys or space to change slides. You get a full report when you stop.
      </p>
      <label className="block text-sm">
        Deck
        <select className={field} value={props.deckId} onChange={e => props.setDeckId(e.target.value)} aria-label="Deck">
          <option value="">No deck (free talk)</option>
          {props.decks.map(d => (
            <option key={d.id} value={d.id}>
              {d.title} ({d.slide_count} slides)
            </option>
          ))}
        </select>
      </label>
      {props.deck && props.deck.slides[0] && (
        <div className="w-64">
          <SlideView deckId={props.deck.id} slide={props.deck.slides[0]} />
        </div>
      )}
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="block text-sm">
          Target length (minutes)
          <input type="number" min={0.5} max={60} step={0.5} className={field} value={props.minutes} onChange={e => props.setMinutes(Number(e.target.value))} aria-label="Target length" />
        </label>
        <label className="block text-sm">
          Name this take (optional)
          <input className={field} value={props.title} onChange={e => props.setTitle(e.target.value)} placeholder="e.g. Demo day run 3" aria-label="Take name" />
        </label>
      </div>
      <label className="flex items-center gap-2 text-sm">
        <input type="checkbox" checked={props.camera} onChange={e => props.setCamera(e.target.checked)} /> Use the camera for eye contact and posture (processed on this computer only)
      </label>
      {props.error && <p className="text-sm text-rose-400" role="alert">{props.error}</p>}
      <button onClick={props.onStart} disabled={props.starting} className="flex items-center gap-2 rounded-xl bg-purple-600 px-5 py-2.5 font-semibold hover:bg-purple-500 disabled:opacity-50">
        <Play className="h-4 w-4" /> {props.starting ? 'Starting...' : 'Start rehearsal'}
      </button>
    </div>
  );
}

function Presenter({
  visible,
  deck,
  camera,
  videoRef,
  targetSeconds,
  take,
  onStop,
}: {
  visible: boolean;
  deck?: Deck;
  camera: boolean;
  videoRef: React.MutableRefObject<HTMLVideoElement | null>;
  targetSeconds: number;
  take: ReturnType<typeof useLiveTake>;
  onStop: () => void;
}) {
  const [index, setIndex] = useState(0);
  const [showNotes, setShowNotes] = useState(true);
  const [elapsed, setElapsed] = useState(0);
  const [slideStart, setSlideStart] = useState(0);
  const slides = deck?.slides ?? [];
  const live = take.phase === 'live';

  useEffect(() => {
    if (!live) return;
    setIndex(0);
    setSlideStart(0);
    const t = setInterval(() => setElapsed(take.session.current?.elapsed ?? 0), 250);
    return () => clearInterval(t);
  }, [live, take.session]);

  const go = useCallback(
    (next: number) => {
      if (!slides.length) return;
      const clamped = Math.max(0, Math.min(slides.length - 1, next));
      if (clamped === index) return;
      setIndex(clamped);
      setSlideStart(take.session.current?.elapsed ?? 0);
      take.slide(clamped);
    },
    [index, slides.length, take],
  );

  useEffect(() => {
    if (!live) return;
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'TEXTAREA') return;
      if (['ArrowRight', 'PageDown', ' '].includes(e.key)) {
        e.preventDefault();
        go(index + 1);
      } else if (['ArrowLeft', 'PageUp'].includes(e.key)) {
        e.preventDefault();
        go(index - 1);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [live, go, index]);

  const slide = slides[index];
  const budget = slide?.time_budget_sec ?? (slides.length ? targetSeconds / slides.length : null);
  const onSlide = elapsed - slideStart;
  const overSlide = budget ? onSlide > budget : false;
  const overTotal = elapsed > targetSeconds;
  const progressPct = useMemo(() => Math.min(100, (elapsed / targetSeconds) * 100), [elapsed, targetSeconds]);

  return (
    <div className={visible ? 'grid h-full grid-cols-1 gap-3 overflow-y-auto p-3 xl:grid-cols-[minmax(0,1fr)_360px]' : 'hidden'}>
      <section className="flex min-h-0 flex-col gap-3">
        <header className="flex flex-wrap items-center gap-3 rounded-2xl bg-neutral-900/70 px-4 py-2">
          <span className="text-sm text-neutral-400" data-testid="slide-counter">
            {slides.length ? `Slide ${index + 1} / ${slides.length}` : 'Free talk'}
          </span>
          {slides.length > 0 && (
            <span className={`font-mono text-sm ${overSlide ? 'text-rose-300' : 'text-neutral-200'}`} title="Time on this slide">
              {fmt(onSlide)}
              {budget ? ` / ${fmt(budget)}` : ''}
            </span>
          )}
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/10" aria-hidden>
            <div className={`h-full ${overTotal ? 'bg-rose-500' : 'bg-purple-500'}`} style={{ width: `${progressPct}%` }} />
          </div>
          <span className={`font-mono text-sm ${overTotal ? 'text-rose-300' : ''}`} data-testid="total-timer">
            {fmt(elapsed)} / {fmt(targetSeconds)}
          </span>
          {camera && live && (
            <button
              onClick={async () => {
                toast('Look straight into the camera lens for 3 seconds...');
                const ok = await take.calibrate();
                toast(ok ? 'Eye contact calibrated.' : 'No face found; calibration skipped.', ok ? 'info' : 'error');
              }}
              className="rounded-lg p-1.5 text-neutral-400 hover:bg-white/5"
              title="Calibrate eye contact"
              aria-label="Calibrate eye contact"
            >
              <Crosshair className="h-4 w-4" />
            </button>
          )}
          <button onClick={onStop} disabled={!live} className="flex items-center gap-1.5 rounded-xl bg-rose-600 px-3 py-1.5 text-sm font-semibold hover:bg-rose-500 disabled:opacity-50">
            <Square className="h-4 w-4" /> Stop
          </button>
        </header>

        {slide && deck ? (
          <div className="flex min-h-0 flex-col gap-2">
            <div className="relative">
              <SlideView deckId={deck.id} slide={slide} />
              <button onClick={() => go(index - 1)} disabled={index === 0} aria-label="Previous slide" className="absolute left-2 top-1/2 -translate-y-1/2 rounded-full bg-black/50 p-2 disabled:opacity-0">
                <ChevronLeft className="h-5 w-5" />
              </button>
              <button onClick={() => go(index + 1)} disabled={index >= slides.length - 1} aria-label="Next slide" className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full bg-black/50 p-2 disabled:opacity-0">
                <ChevronRight className="h-5 w-5" />
              </button>
            </div>
            <div className="grid gap-2 md:grid-cols-[200px_minmax(0,1fr)]">
              <div>
                <span className="text-[11px] uppercase tracking-wider text-neutral-500">Next</span>
                {slides[index + 1] ? <SlideView deckId={deck.id} slide={slides[index + 1]} compact /> : <p className="text-xs text-neutral-500">Last slide</p>}
              </div>
              <div className="rounded-2xl bg-neutral-900/70 p-3">
                <button onClick={() => setShowNotes(s => !s)} className="flex items-center gap-1 text-[11px] uppercase tracking-wider text-neutral-500 hover:text-white" aria-pressed={showNotes}>
                  <StickyNote className="h-3.5 w-3.5" /> Teleprompter {showNotes ? 'on' : 'off'}
                </button>
                {showNotes && <p className="mt-1 whitespace-pre-wrap text-base leading-relaxed text-neutral-200">{slide.notes || 'No notes for this slide.'}</p>}
              </div>
            </div>
          </div>
        ) : (
          <div className="min-h-[300px] flex-1">
            <LiveCaptions placeholder="Free talk: just start speaking." />
          </div>
        )}
      </section>

      <aside className="space-y-3">
        <CameraView ref={videoRef} overlay={false} active={live && camera} />
        <Hud />
        {slide && (
          <div className="h-40">
            <LiveCaptions />
          </div>
        )}
      </aside>
    </div>
  );
}
