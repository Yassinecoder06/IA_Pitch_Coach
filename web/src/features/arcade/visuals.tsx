import React from 'react';
import { useSensors } from '@/stores/sensors';
import type { EyeLock, FillerNinja, Game, GameId, PaceRacer, PausePower, PitchSurfer, Statue } from './games';

/** One visual per game; each reads the game's state (and the live sensors for context). */
export function GameVisual({ id, game }: { id: GameId; game: Game<any> }) {
  switch (id) {
    case 'filler_ninja':
      return <NinjaView g={game as FillerNinja} />;
    case 'pitch_surfer':
      return <SurferView g={game as PitchSurfer} />;
    case 'eye_lock':
      return <EyeLockView g={game as EyeLock} />;
    case 'statue':
      return <StatueView g={game as Statue} />;
    case 'pace_racer':
      return <PaceView g={game as PaceRacer} />;
    case 'pause_power':
      return <PauseView g={game as PausePower} />;
  }
}

function NinjaView({ g }: { g: FillerNinja }) {
  const fillers = useSensors(s => s.fillers);
  const last = fillers[fillers.length - 1];
  const multiplier = 1 + Math.floor(g.state.combo / 10);
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 p-6" data-testid="ninja">
      <div className="text-sm uppercase tracking-widest text-neutral-500">Combo</div>
      <div className="text-8xl font-black text-purple-300">{g.state.combo}</div>
      <div className="text-sm text-neutral-400">x{multiplier} points · {g.state.fillers} fillers sliced your combo</div>
      {last && <div className="mt-4 animate-pulse rounded-full bg-rose-500/20 px-4 py-1 text-rose-300">Last filler: “{last.filler}”</div>}
    </div>
  );
}

function SurferView({ g }: { g: PitchSurfer }) {
  const trail = useSensors(s => s.voice?.pitchTrail ?? []);
  const W = 600;
  const H = 300;
  const voiced = trail.filter(v => v > 0);
  const mean = voiced.length ? voiced.reduce((a, b) => a + b, 0) / voiced.length : 180;
  // The wave is your own pitch over the last 3 s; the sea level rises when you go flat.
  const pts = trail.map((v, i) => `${(i / Math.max(1, trail.length - 1)) * W},${v > 0 ? H / 2 - ((v - mean) / mean) * 260 : H / 2}`);
  const sea = H - g.state.altitude * (H - 40);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full" role="img" aria-label={`Surfer altitude ${Math.round(g.state.altitude * 100)}%`} data-testid="surfer">
      <polyline points={pts.join(' ')} fill="none" stroke="var(--viz-series-2)" strokeWidth={3} strokeLinejoin="round" />
      <rect x={0} y={sea} width={W} height={H - sea} fill="var(--viz-series-1)" opacity={0.35} />
      <text x={W - 60} y={sea - 10} fontSize={36} textAnchor="middle">
        🏄
      </text>
      <text x={12} y={24} fontSize={14} fill="var(--viz-text-2)">
        Wipeouts: {g.state.wipeouts}
      </text>
    </svg>
  );
}

function Ring({ value, label, color }: { value: number; label: string; color: string }) {
  const r = 90;
  const c = 2 * Math.PI * r;
  return (
    <svg viewBox="0 0 220 220" className="h-64 w-64" role="img" aria-label={`${label} ${Math.round(value)}%`}>
      <circle cx={110} cy={110} r={r} fill="none" stroke="var(--viz-grid)" strokeWidth={16} />
      <circle
        cx={110}
        cy={110}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={16}
        strokeLinecap="round"
        strokeDasharray={`${(value / 100) * c} ${c}`}
        transform="rotate(-90 110 110)"
      />
      <text x={110} y={108} textAnchor="middle" fontSize={40} fontWeight={800} fill="var(--viz-text)">
        {Math.round(value)}%
      </text>
      <text x={110} y={138} textAnchor="middle" fontSize={13} fill="var(--viz-text-2)">
        {label}
      </text>
    </svg>
  );
}

function EyeLockView({ g }: { g: EyeLock }) {
  const state = useSensors(s => (s.gaze?.facePresent ? s.gaze.state : 'lost'));
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2" data-testid="eyelock">
      <Ring value={g.state.charge} label="Charging" color="var(--viz-series-3)" />
      <div className="text-sm text-neutral-400">
        Locks: <span className="font-bold text-white">{g.state.locks}</span> · eye contact now: {state === 'good' ? 'on the lens' : state === 'weak' ? 'drifting' : 'away'}
      </div>
    </div>
  );
}

function StatueView({ g }: { g: Statue }) {
  const issues = useSensors(s => s.posture?.issues ?? []);
  const fidget = useSensors(s => s.motion?.fidget ?? 0);
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2" data-testid="statue">
      <Ring value={Math.min(100, (g.state.streak / 10) * 100)} label={`Steady ${Math.round(g.state.streak)}s`} color="var(--viz-series-1)" />
      <div className="text-sm text-neutral-400">
        Breaks: {g.state.breaks} · fidget {Math.round(fidget * 100)}% {issues.length ? `· ${issues.join(', ').replace(/_/g, ' ')}` : ''}
      </div>
    </div>
  );
}

function PaceView({ g }: { g: PaceRacer }) {
  const W = 400;
  const H = 230;
  const cx = W / 2;
  const cy = 200;
  const r = 160;
  const min = 60;
  const max = 240;
  const angle = (wpm: number) => Math.PI * (1 - (Math.min(max, Math.max(min, wpm)) - min) / (max - min));
  const arc = (a0: number, a1: number) => {
    const p = (a: number) => `${cx + r * Math.cos(a)},${cy - r * Math.sin(a)}`;
    return `M${p(a0)} A${r},${r} 0 0 1 ${p(a1)}`;
  };
  const needle = angle(g.state.wpm || min);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full" role="img" aria-label={`Pace ${Math.round(g.state.wpm)} words per minute`} data-testid="pace">
      <path d={arc(Math.PI, 0)} fill="none" stroke="var(--viz-grid)" strokeWidth={22} />
      <path d={arc(angle(g.band[0]), angle(g.band[1]))} fill="none" stroke="var(--viz-series-3)" strokeWidth={22} />
      <line x1={cx} y1={cy} x2={cx + (r - 30) * Math.cos(needle)} y2={cy - (r - 30) * Math.sin(needle)} stroke="var(--viz-text)" strokeWidth={4} strokeLinecap="round" />
      <text x={cx} y={cy - 40} textAnchor="middle" fontSize={44} fontWeight={800} fill="var(--viz-text)">
        {g.state.wpm ? Math.round(g.state.wpm) : '--'}
      </text>
      <text x={cx} y={cy - 16} textAnchor="middle" fontSize={13} fill="var(--viz-text-2)">
        wpm · green {g.band[0]}-{g.band[1]}
      </text>
    </svg>
  );
}

function PauseView({ g }: { g: PausePower }) {
  const cue = g.state.phase === 'cue';
  const pausing = g.state.phase === 'pausing';
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3" data-testid="pause">
      <div className={`text-7xl font-black ${cue ? 'animate-pulse text-amber-300' : pausing ? 'text-sky-300' : 'text-neutral-700'}`}>
        {cue ? 'PAUSE' : pausing ? `${(g.state.pauseMs / 1000).toFixed(1)}s` : 'Keep talking'}
      </div>
      <div className="text-sm text-neutral-400">
        Aim for {g.window[0]}-{g.window[1]} s · hits {g.state.hits} · misses {g.state.misses}
      </div>
    </div>
  );
}
