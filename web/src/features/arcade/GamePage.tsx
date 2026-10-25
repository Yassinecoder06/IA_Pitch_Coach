import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Play, RotateCcw, Trophy } from 'lucide-react';
import { arcadeApi } from '@/lib/arcade';
import { liveWpm, useSensors } from '@/stores/sensors';
import { CameraView } from '@/features/rehearse/components/CameraView';
import { LiveCaptions } from '@/features/rehearse/components/LiveCaptions';
import { useLiveTake } from '@/features/rehearse/useLiveTake';
import { toast } from '@/stores/toast';
import { GAMES, PROMPTS, type Difficulty, type Game, type GameId } from './games';
import { GameVisual } from './visuals';

type Phase = 'setup' | 'countdown' | 'playing' | 'over';

export function GamePage() {
  const { gameId = '' } = useParams();
  const info = GAMES[gameId as GameId];
  const qc = useQueryClient();
  const take = useLiveTake();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [difficulty, setDifficulty] = useState<Difficulty>('medium');
  const [phase, setPhase] = useState<Phase>('setup');
  const [count, setCount] = useState(3);
  const [, setTick] = useState(0); // re-render every frame while playing
  const [result, setResult] = useState<{ score: number; best: boolean; previous: number | null; stats: Record<string, number> } | null>(null);
  const game = useRef<Game<any> | null>(null);
  const prompt = useMemo(() => PROMPTS[Math.floor(Math.random() * PROMPTS.length)], [phase === 'setup']); // eslint-disable-line react-hooks/exhaustive-deps

  // Animation loop: feed the game from the sensor store.
  useEffect(() => {
    if (phase !== 'playing') return;
    let raf = 0;
    let last = performance.now();
    let seenWords = useSensors.getState().words.length;
    let seenFillers = useSensors.getState().fillers.length;
    const loop = (now: number) => {
      const s = useSensors.getState();
      const g = game.current!;
      g.step({
        dt: (now - last) / 1000,
        voice: s.voice,
        gaze: s.gaze,
        posture: s.posture,
        motion: s.motion,
        wpm: liveWpm(s.words, 20),
        newWords: s.words.length - seenWords,
        newFillers: s.fillers.length - seenFillers,
      });
      seenWords = s.words.length;
      seenFillers = s.fillers.length;
      last = now;
      setTick(t => t + 1);
      if (g.done) void finish();
      else raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [phase]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!info) return <p className="p-6 text-sm text-rose-400">Unknown game.</p>;

  async function start() {
    game.current = info.create(difficulty);
    setResult(null);
    const s = await take.start({ camera: info.needs.camera, mic: info.needs.mic, video: videoRef.current });
    if (!s) return;
    setPhase('countdown');
    for (let i = 3; i > 0; i--) {
      setCount(i);
      await new Promise(r => setTimeout(r, 1000));
    }
    setPhase('playing');
  }

  async function finish() {
    setPhase('over');
    const g = game.current!;
    await take.stop();
    try {
      const res = await arcadeApi.record({ game: info.id, difficulty, score: g.score, stats: g.stats() });
      setResult({ score: g.score, best: res.personal_best, previous: res.previous_best, stats: g.stats() });
      qc.invalidateQueries({ queryKey: ['arcade'] });
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Could not save the run', 'error');
      setResult({ score: g.score, best: false, previous: null, stats: g.stats() });
    }
  }

  const g = game.current;
  const remaining = g ? Math.max(0, Math.ceil(g.duration - g.elapsed)) : 0;

  return (
    <div className="grid h-full grid-cols-1 gap-4 overflow-y-auto p-6 lg:grid-cols-[minmax(0,1fr)_340px]">
      <section className="flex flex-col gap-4">
        <header className="flex items-center justify-between">
          <div>
            <Link to="/arcade" className="text-xs text-neutral-500 hover:text-white">
              ← Arcade
            </Link>
            <h1 className="text-xl font-bold">{info.title}</h1>
            <p className="text-sm text-neutral-400">{info.how}</p>
          </div>
          {phase === 'playing' && g && (
            <div className="text-right">
              <div className="font-mono text-3xl font-black" data-testid="game-score">{Math.round(g.score)}</div>
              <div className="text-xs text-neutral-500" data-testid="game-time">{remaining}s left</div>
            </div>
          )}
        </header>

        {(info.needs.captions || info.id === 'pause_power') && phase !== 'setup' && (
          <p className="rounded-xl bg-neutral-900/70 px-4 py-2 text-sm">
            <span className="text-neutral-500">Talk about: </span>
            {prompt}
          </p>
        )}

        <div className="relative min-h-[320px] flex-1 overflow-hidden rounded-2xl border border-white/10 bg-neutral-950" data-testid="game-stage">
          {g && (phase === 'playing' || phase === 'over') && <GameVisual id={info.id} game={g} />}
          {phase === 'countdown' && (
            <div className="absolute inset-0 flex items-center justify-center text-7xl font-black text-purple-300" aria-live="assertive">
              {count}
            </div>
          )}
          {phase === 'setup' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
              <div className="flex gap-2" role="radiogroup" aria-label="Difficulty">
                {(['easy', 'medium', 'hard'] as Difficulty[]).map(d => (
                  <button
                    key={d}
                    role="radio"
                    aria-checked={difficulty === d}
                    onClick={() => setDifficulty(d)}
                    className={`rounded-xl px-4 py-2 text-sm capitalize ${difficulty === d ? 'bg-purple-600 text-white' : 'border border-white/10 text-neutral-300'}`}
                  >
                    {d}
                  </button>
                ))}
              </div>
              <button onClick={() => void start()} disabled={take.phase === 'starting'} className="flex items-center gap-2 rounded-xl bg-purple-600 px-6 py-3 font-semibold hover:bg-purple-500 disabled:opacity-50">
                <Play className="h-5 w-5" /> Play
              </button>
              {take.error && <p className="text-sm text-rose-400">{take.error}</p>}
            </div>
          )}
          {phase === 'over' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/70" data-testid="game-over">
              {!result ? (
                <p>Saving...</p>
              ) : (
                <>
                  {result.best && (
                    <p className="flex items-center gap-1 text-amber-300">
                      <Trophy className="h-5 w-5" /> New personal best!
                    </p>
                  )}
                  <div className="text-6xl font-black">{Math.round(result.score)}</div>
                  {result.previous != null && !result.best && <p className="text-sm text-neutral-400">Best: {Math.round(result.previous)}</p>}
                  <dl className="grid grid-cols-2 gap-x-4 text-sm">
                    {Object.entries(result.stats).map(([k, v]) => (
                      <React.Fragment key={k}>
                        <dt className="text-neutral-400">{k.replace(/_/g, ' ')}</dt>
                        <dd className="text-right font-mono">{v}</dd>
                      </React.Fragment>
                    ))}
                  </dl>
                  <div className="flex gap-2">
                    <button onClick={() => setPhase('setup')} className="flex items-center gap-1.5 rounded-xl bg-purple-600 px-4 py-2 font-semibold hover:bg-purple-500">
                      <RotateCcw className="h-4 w-4" /> Play again
                    </button>
                    <Link to="/arcade" className="rounded-xl border border-white/10 px-4 py-2">
                      Back to arcade
                    </Link>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </section>

      <aside className="space-y-3">
        <div className={info.needs.camera ? '' : 'hidden'}>
          <CameraView ref={videoRef} overlay active={take.phase === 'live' && info.needs.camera} />
        </div>
        {info.needs.captions && (
          <div className="h-64">
            <LiveCaptions />
          </div>
        )}
      </aside>
    </div>
  );
}
