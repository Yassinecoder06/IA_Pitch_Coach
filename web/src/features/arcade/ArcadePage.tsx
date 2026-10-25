import React from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Camera, Flame, Mic, Trophy } from 'lucide-react';
import { arcadeApi } from '@/lib/arcade';
import { GAMES } from './games';

export function ArcadePage() {
  const { data } = useQuery({ queryKey: ['arcade'], queryFn: arcadeApi.summary });

  return (
    <div className="h-full space-y-6 overflow-y-auto p-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold">Training arcade</h1>
          <p className="text-sm text-neutral-400">One-minute drills that build the habits your rehearsals measure.</p>
        </div>
        {data && (
          <div className="flex items-center gap-2 rounded-2xl bg-neutral-900/70 px-4 py-2" data-testid="streak">
            <Flame className={`h-5 w-5 ${data.streak_days ? 'text-orange-400' : 'text-neutral-600'}`} />
            <span className="font-bold">{data.streak_days}</span>
            <span className="text-sm text-neutral-400">day streak{data.played_today ? '' : ' · play today to keep it'}</span>
          </div>
        )}
      </header>

      {data && (
        <section className="rounded-2xl border border-purple-500/30 bg-purple-950/20 p-4" data-testid="daily-queue">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-purple-200">Today's drills</h2>
          <ol className="mt-2 grid gap-2 md:grid-cols-3">
            {data.queue.map((q, i) => (
              <li key={q.game}>
                <Link to={`/arcade/${q.game}`} className="block rounded-xl bg-neutral-950/60 p-3 hover:bg-neutral-900">
                  <div className="font-semibold">
                    {i + 1}. {GAMES[q.game].title}
                  </div>
                  <div className="text-xs text-neutral-400">Because {q.reason}.</div>
                </Link>
              </li>
            ))}
          </ol>
        </section>
      )}

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {Object.values(GAMES).map(g => {
          const bests = data?.games.find(x => x.id === g.id)?.bests ?? {};
          const best = Math.max(0, ...Object.values(bests).map(Number));
          return (
            <Link key={g.id} to={`/arcade/${g.id}`} className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4 hover:border-purple-500/40">
              <div className="flex items-center justify-between">
                <h2 className="font-semibold">{g.title}</h2>
                <span className="flex gap-1 text-neutral-500">
                  {g.needs.mic && <Mic className="h-4 w-4" aria-label="Uses the microphone" />}
                  {g.needs.camera && <Camera className="h-4 w-4" aria-label="Uses the camera" />}
                </span>
              </div>
              <p className="text-xs uppercase tracking-wider text-purple-300">{g.skill}</p>
              <p className="mt-1 text-sm text-neutral-400">{g.how}</p>
              {best > 0 && (
                <p className="mt-2 flex items-center gap-1 text-xs text-amber-200">
                  <Trophy className="h-3.5 w-3.5" /> Best {Math.round(best)}
                </p>
              )}
            </Link>
          );
        })}
      </section>
    </div>
  );
}
