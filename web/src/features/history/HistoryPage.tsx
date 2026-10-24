import React, { useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ArrowLeftRight, CheckCircle2, CircleDashed, Target, Trash2, Trophy, XCircle } from 'lucide-react';
import { decksApi } from '@/lib/decks';
import { FACT_INFO, rehearsalsApi, SCORE_LABELS, type Goal } from '@/lib/rehearsals';
import { TrendChart } from '@/components/ui/charts';
import { toast } from '@/stores/toast';

const fmt = (s: number | null) => (s == null ? '--' : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`);

export function HistoryPage() {
  const navigate = useNavigate();
  const [deckId, setDeckId] = useState('');
  const [picked, setPicked] = useState<string[]>([]);
  const decks = useQuery({ queryKey: ['decks'], queryFn: decksApi.list });
  const list = useQuery({ queryKey: ['rehearsals', deckId], queryFn: () => rehearsalsApi.list(deckId || undefined) });
  const deckTitle = useMemo(() => Object.fromEntries((decks.data?.decks ?? []).map(d => [d.id, d.title])), [decks.data]);

  const done = useMemo(() => (list.data?.rehearsals ?? []).filter(r => r.status === 'done').slice().reverse(), [list.data]);
  const best = useMemo(() => done.reduce<(typeof done)[number] | null>((b, r) => ((r.overall_score ?? -1) > (b?.overall_score ?? -1) ? r : b), null), [done]);

  function toggle(id: string) {
    setPicked(p => (p.includes(id) ? p.filter(x => x !== id) : [...p, id].slice(-2)));
  }

  return (
    <div className="h-full space-y-6 overflow-y-auto p-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold">History</h1>
          <p className="text-sm text-neutral-400">Every rehearsal, how you are trending, and whether you are hitting your goals.</p>
        </div>
        <select
          value={deckId}
          onChange={e => {
            setDeckId(e.target.value);
            setPicked([]);
          }}
          aria-label="Filter by deck"
          className="rounded-xl border border-white/10 bg-neutral-900 px-3 py-2 text-sm"
        >
          <option value="">All decks</option>
          {(decks.data?.decks ?? []).map(d => (
            <option key={d.id} value={d.id}>
              {d.title}
            </option>
          ))}
        </select>
      </header>

      <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4" data-testid="trend">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-neutral-400">Trend</h2>
            {best && (
              <Link to={`/report/${best.id}`} className="flex items-center gap-1 text-xs text-amber-200">
                <Trophy className="h-3.5 w-3.5" /> Best: {best.title} ({Math.round(best.overall_score ?? 0)})
              </Link>
            )}
          </div>
          <TrendChart
            labels={done.map(r => new Date(r.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }))}
            series={[
              { key: 'overall', label: 'Overall', color: 'var(--viz-series-1)', values: done.map(r => r.overall_score) },
              { key: 'delivery', label: 'Delivery', color: 'var(--viz-series-2)', values: done.map(r => r.scores?.delivery ?? null) },
              { key: 'content', label: 'Content', color: 'var(--viz-series-3)', values: done.map(r => r.scores?.content ?? null) },
            ]}
            onSelect={i => navigate(`/report/${done[i].id}`)}
          />
        </div>
        <GoalsPanel />
      </section>

      <section className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-neutral-400">Attempts</h2>
          <button
            disabled={picked.length !== 2}
            onClick={() => {
              const [x, y] = [...picked].sort(
                (p, q) => +new Date(list.data!.rehearsals.find(r => r.id === p)!.created_at) - +new Date(list.data!.rehearsals.find(r => r.id === q)!.created_at),
              );
              navigate(`/history/compare?a=${x}&b=${y}`);
            }}
            className="flex items-center gap-1.5 rounded-xl bg-purple-600 px-3 py-1.5 text-sm font-semibold hover:bg-purple-500 disabled:opacity-40"
          >
            <ArrowLeftRight className="h-4 w-4" /> Compare selected ({picked.length}/2)
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm" data-testid="attempts">
            <thead className="text-left text-xs text-neutral-500">
              <tr>
                <th className="w-8" />
                <th className="py-1">Take</th>
                <th>Deck</th>
                <th>Date</th>
                <th>Length</th>
                <th className="text-right">Overall</th>
                <th className="text-right">Delivery</th>
                <th className="text-right">Content</th>
              </tr>
            </thead>
            <tbody>
              {(list.data?.rehearsals ?? []).map(r => (
                <tr key={r.id} className="border-t border-white/5">
                  <td>
                    <input
                      type="checkbox"
                      checked={picked.includes(r.id)}
                      disabled={r.status !== 'done'}
                      onChange={() => toggle(r.id)}
                      aria-label={`Select ${r.title} to compare`}
                    />
                  </td>
                  <td className="py-2">
                    <Link to={`/report/${r.id}`} className="hover:underline">
                      {r.title}
                    </Link>
                    {r.status !== 'done' && <span className="ml-2 text-xs text-neutral-500">{r.status}</span>}
                    {best?.id === r.id && <Trophy className="ml-1 inline h-3.5 w-3.5 text-amber-300" aria-label="Best attempt" />}
                  </td>
                  <td className="text-neutral-400">{r.deck_id ? deckTitle[r.deck_id] ?? 'Deleted deck' : 'Free talk'}</td>
                  <td className="text-neutral-400">{new Date(r.created_at).toLocaleString()}</td>
                  <td className="font-mono">{fmt(r.duration_sec)}</td>
                  <td className="text-right font-mono">{r.overall_score == null ? '--' : Math.round(r.overall_score)}</td>
                  <td className="text-right font-mono">{r.scores?.delivery == null ? '--' : Math.round(r.scores.delivery)}</td>
                  <td className="text-right font-mono">{r.scores?.content == null ? '--' : Math.round(r.scores.content)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {list.data?.rehearsals.length === 0 && (
            <p className="p-6 text-center text-sm text-neutral-500">
              No rehearsals yet. <Link to="/rehearse" className="underline">Start one</Link>.
            </p>
          )}
        </div>
      </section>
    </div>
  );
}

const OP_TEXT = { gte: 'at least', lte: 'at most', between: 'between' };

export function goalLabel(metric: string) {
  return FACT_INFO[metric] ? `${FACT_INFO[metric].label} (${FACT_INFO[metric].unit || 'score'})` : `${SCORE_LABELS[metric] ?? metric} score`;
}

function GoalsPanel() {
  const qc = useQueryClient();
  const { data } = useQuery({ queryKey: ['goals'], queryFn: rehearsalsApi.goals });
  const [metric, setMetric] = useState('eye_contact_pct');
  const [op, setOp] = useState<'gte' | 'lte' | 'between'>('gte');
  const [target, setTarget] = useState(70);
  const [targetMax, setTargetMax] = useState(160);
  const add = useMutation({
    mutationFn: () => rehearsalsApi.addGoal({ metric, op, target, target_max: op === 'between' ? targetMax : null }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['goals'] }),
    onError: (e: Error) => toast(e.message, 'error'),
  });
  const remove = useMutation({
    mutationFn: (id: string) => rehearsalsApi.deleteGoal(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['goals'] }),
  });
  const field = 'rounded-lg border border-white/10 bg-neutral-900 px-2 py-1.5 text-xs';

  return (
    <div className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4" data-testid="goals">
      <h2 className="mb-2 flex items-center gap-1.5 text-sm font-semibold uppercase tracking-wider text-neutral-400">
        <Target className="h-4 w-4" /> Goals
      </h2>
      <ul className="space-y-2">
        {(data?.goals ?? []).map((g: Goal) => (
          <li key={g.id} className="flex items-start gap-2 rounded-xl bg-neutral-950/60 p-2 text-sm">
            {g.status === 'met' ? (
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" aria-label="Met" />
            ) : g.status === 'not_met' ? (
              <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" aria-label="Not met" />
            ) : (
              <CircleDashed className="mt-0.5 h-4 w-4 shrink-0 text-neutral-500" aria-label="No data yet" />
            )}
            <div className="flex-1">
              <div>
                {goalLabel(g.metric)} {OP_TEXT[g.op]} {g.target}
                {g.op === 'between' ? `-${g.target_max}` : ''}
              </div>
              <div className="text-xs text-neutral-500">
                {g.status === 'no_data' ? 'No data yet' : `Latest ${g.latest} · last 3 avg ${g.recent_avg}`} · {g.status === 'met' ? 'met' : g.status === 'not_met' ? 'not met' : ''}
              </div>
            </div>
            <button onClick={() => remove.mutate(g.id)} aria-label="Delete goal" className="text-neutral-500 hover:text-rose-300">
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </li>
        ))}
        {data?.goals.length === 0 && <li className="text-xs text-neutral-500">Set a target, e.g. eye contact at least 70%.</li>}
      </ul>
      <form
        className="mt-3 flex flex-wrap items-center gap-1.5"
        onSubmit={e => {
          e.preventDefault();
          add.mutate();
        }}
      >
        <select className={field} value={metric} onChange={e => setMetric(e.target.value)} aria-label="Goal metric">
          {(data?.metrics ?? []).map(m => (
            <option key={m} value={m}>
              {goalLabel(m)}
            </option>
          ))}
        </select>
        <select className={field} value={op} onChange={e => setOp(e.target.value as typeof op)} aria-label="Goal condition">
          <option value="gte">at least</option>
          <option value="lte">at most</option>
          <option value="between">between</option>
        </select>
        <input className={`${field} w-16`} type="number" value={target} onChange={e => setTarget(Number(e.target.value))} aria-label="Target" />
        {op === 'between' && <input className={`${field} w-16`} type="number" value={targetMax} onChange={e => setTargetMax(Number(e.target.value))} aria-label="Upper target" />}
        <button className="rounded-lg bg-purple-600 px-2.5 py-1.5 text-xs font-semibold hover:bg-purple-500">Add goal</button>
      </form>
    </div>
  );
}
