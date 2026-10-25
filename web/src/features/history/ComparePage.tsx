import React, { useEffect, useMemo } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ArrowDownRight, ArrowUpRight, CheckCircle2, Minus, PlusCircle, RefreshCcw } from 'lucide-react';
import { FACT_INFO, rehearsalsApi, SCORE_LABELS } from '@/lib/rehearsals';

/** "12 st", "167 wpm", "3/min" but "94%" and "13s". */
const withUnit = (v: number, unit = '') => `${Math.round(v * 10) / 10}${/^[a-z]/i.test(unit) && unit !== 's' ? ' ' : ''}${unit}`;

/** ▲ / ▼ with icon, sign and colour, so direction never depends on colour alone. */
function Delta({ value, better = 'up', unit = '' }: { value: number | null; better?: 'up' | 'down' | null; unit?: string }) {
  if (value == null) return <span className="text-neutral-600">--</span>;
  if (Math.abs(value) < 0.05) return <span className="inline-flex items-center gap-0.5 text-neutral-400"><Minus className="h-3 w-3" /> 0</span>;
  const improved = better === null ? null : better === 'up' ? value > 0 : value < 0;
  const cls = improved === null ? 'text-neutral-300' : improved ? 'text-emerald-300' : 'text-rose-300';
  const Icon = value > 0 ? ArrowUpRight : ArrowDownRight;
  return (
    <span className={`inline-flex items-center gap-0.5 font-mono ${cls}`} aria-label={`${value > 0 ? 'up' : 'down'} ${Math.abs(value)}${unit}${improved === null ? '' : improved ? ', better' : ', worse'}`}>
      <Icon className="h-3.5 w-3.5" />
      {value > 0 ? '+' : ''}
      {withUnit(value, unit)}
    </span>
  );
}

export function ComparePage() {
  const [params, setParams] = useSearchParams();
  const list = useQuery({ queryKey: ['rehearsals', ''], queryFn: () => rehearsalsApi.list() });
  const done = useMemo(() => (list.data?.rehearsals ?? []).filter(r => r.status === 'done'), [list.data]);
  const a = params.get('a') ?? '';
  const b = params.get('b') ?? '';

  // From a report's "Compare" button only b is known: default a to the previous take of the same deck.
  useEffect(() => {
    if (a || !b || !done.length) return;
    const current = done.find(r => r.id === b);
    const older = done.filter(r => r.id !== b && r.deck_id === current?.deck_id && +new Date(r.created_at) < +new Date(current!.created_at));
    const pick = older[0] ?? done.find(r => r.id !== b);
    if (pick) setParams({ a: pick.id, b }, { replace: true });
  }, [a, b, done, setParams]);

  const cmp = useQuery({ queryKey: ['compare', a, b], queryFn: () => rehearsalsApi.compare(a, b), enabled: !!a && !!b && a !== b });
  const choose = (key: 'a' | 'b') => (e: React.ChangeEvent<HTMLSelectElement>) => setParams({ a, b, [key]: e.target.value });
  const select = 'rounded-xl border border-white/10 bg-neutral-900 px-3 py-2 text-sm';

  return (
    <div className="h-full space-y-6 overflow-y-auto p-6">
      <header>
        <Link to="/history" className="text-xs text-neutral-500 hover:text-white">
          ← History
        </Link>
        <h1 className="text-xl font-bold">Compare attempts</h1>
      </header>
      <div className="flex flex-wrap items-center gap-2">
        <select className={select} value={a} onChange={choose('a')} aria-label="Attempt A">
          <option value="">Choose attempt A</option>
          {done.map(r => (
            <option key={r.id} value={r.id}>
              {r.title} · {new Date(r.created_at).toLocaleDateString()}
            </option>
          ))}
        </select>
        <span className="text-neutral-500">→</span>
        <select className={select} value={b} onChange={choose('b')} aria-label="Attempt B">
          <option value="">Choose attempt B</option>
          {done.map(r => (
            <option key={r.id} value={r.id}>
              {r.title} · {new Date(r.created_at).toLocaleDateString()}
            </option>
          ))}
        </select>
      </div>
      {a && a === b && <p className="text-sm text-amber-300">Pick two different attempts.</p>}
      {cmp.error && <p className="text-sm text-rose-400">{(cmp.error as Error).message}</p>}

      {cmp.data && (
        <>
          <section className="grid gap-4 lg:grid-cols-2">
            <div className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4">
              <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-neutral-400">Scores (higher is better)</h2>
              <table className="w-full text-sm" data-testid="compare-scores">
                <thead className="text-left text-xs text-neutral-500">
                  <tr>
                    <th className="py-1">Metric</th>
                    <th className="text-right">A</th>
                    <th className="text-right">B</th>
                    <th className="text-right">Change</th>
                  </tr>
                </thead>
                <tbody>
                  {cmp.data.scores.map(s => (
                    <tr key={s.metric} className="border-t border-white/5">
                      <td className="py-1.5">{SCORE_LABELS[s.metric] ?? s.metric}</td>
                      <td className="text-right font-mono">{s.a == null ? '--' : Math.round(s.a)}</td>
                      <td className="text-right font-mono">{s.b == null ? '--' : Math.round(s.b)}</td>
                      <td className="text-right">
                        <Delta value={s.delta} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4">
              <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-neutral-400">Measurements</h2>
              <table className="w-full text-sm" data-testid="compare-facts">
                <tbody>
                  {cmp.data.facts.map(f => {
                    const info = FACT_INFO[f.metric];
                    return (
                      <tr key={f.metric} className="border-t border-white/5 first:border-0">
                        <td className="py-1.5">{info?.label ?? f.metric}</td>
                        <td className="text-right font-mono">{f.a == null ? '--' : withUnit(f.a, info?.unit)}</td>
                        <td className="text-right font-mono">{f.b == null ? '--' : withUnit(f.b, info?.unit)}</td>
                        <td className="text-right">
                          <Delta value={f.delta} better={info?.better ?? null} unit={info?.unit} />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="mt-2 text-[11px] text-neutral-500">Pace and length have a target band, so their change is shown without a better/worse colour.</p>
            </div>
          </section>

          <section className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4" data-testid="compare-cues">
            <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-neutral-400">Coaching notes</h2>
            <div className="grid gap-4 md:grid-cols-3">
              {(['resolved', 'persisting', 'new'] as const).map(status => {
                const items = cmp.data!.cues.filter(c => c.status === status);
                const Icon = status === 'resolved' ? CheckCircle2 : status === 'persisting' ? RefreshCcw : PlusCircle;
                const tone = status === 'resolved' ? 'text-emerald-300' : status === 'persisting' ? 'text-amber-300' : 'text-rose-300';
                return (
                  <div key={status}>
                    <h3 className={`flex items-center gap-1 text-sm font-semibold capitalize ${tone}`}>
                      <Icon className="h-4 w-4" /> {status} ({items.length})
                    </h3>
                    <ul className="mt-1 space-y-1 text-sm text-neutral-300">
                      {items.map(c => (
                        <li key={c.id}>{c.message}</li>
                      ))}
                      {!items.length && <li className="text-neutral-600">None</li>}
                    </ul>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="grid gap-4 md:grid-cols-2">
            {(['a', 'b'] as const).map(k => (
              <div key={k} className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4">
                <h2 className="text-sm font-semibold">
                  {k.toUpperCase()}:{' '}
                  <Link to={`/report/${cmp.data![k].id}`} className="underline">
                    {cmp.data![k].title}
                  </Link>
                </h2>
                <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-neutral-300">
                  {cmp.data!.priorities[k].map((p, i) => (
                    <li key={i}>{p}</li>
                  ))}
                </ol>
              </div>
            ))}
          </section>
        </>
      )}
    </div>
  );
}
