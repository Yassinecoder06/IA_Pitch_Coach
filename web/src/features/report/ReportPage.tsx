import React, { useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { AlertTriangle, ArrowLeftRight, Info, Loader2, MessageCircleQuestion, Printer, RefreshCw, Trash2, XOctagon } from 'lucide-react';
import { METRIC_LABELS, rehearsalsApi, type Cue, type MetricKey, type Rehearsal } from '@/lib/rehearsals';
import { decksApi } from '@/lib/decks';
import { RadarChart, ScoreBars, TimelineChart, type TimelinePanel } from '@/components/ui/charts';
import { toast } from '@/stores/toast';
import { attachesToPrevious } from '@/lib/words';

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;
const DIMENSIONS = ['clarity', 'structure', 'persuasion', 'delivery', 'confidence', 'relevance'] as const;

export function ReportPage() {
  const { rehearsalId = '' } = useParams();
  const { data: r, isLoading, error } = useQuery({
    queryKey: ['rehearsal', rehearsalId],
    queryFn: () => rehearsalsApi.get(rehearsalId),
    refetchInterval: q => (q.state.data && ['processing', 'recording'].includes(q.state.data.status) ? 2000 : false),
  });

  if (isLoading) return <p className="p-6 text-sm text-neutral-500">Loading report...</p>;
  if (error || !r) return <p className="p-6 text-sm text-rose-400">Report not found.</p>;
  if (r.status === 'processing' || r.status === 'recording' || r.status === 'ready') {
    return (
      <div className="flex h-full items-center justify-center" role="status">
        <div className="text-center">
          <Loader2 className="mx-auto h-8 w-8 animate-spin text-purple-400" />
          <p className="mt-2">{r.status === 'ready' ? 'This take has not been recorded yet.' : `Analysing: ${r.progress ?? 'queued'}...`}</p>
        </div>
      </div>
    );
  }
  if (r.status === 'failed') return <p className="p-6 text-sm text-rose-400">Analysis failed: {r.error}</p>;
  return <Report r={r} />;
}

export function Report({ r, printMode = false }: { r: Rehearsal; printMode?: boolean }) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const audio = useRef<HTMLAudioElement | null>(null);
  const [playhead, setPlayhead] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const a = r.analysis!;
  const deck = useQuery({ queryKey: ['deck', r.deck_id], queryFn: () => decksApi.get(r.deck_id!), enabled: !!r.deck_id });
  const titles = useMemo(() => Object.fromEntries((deck.data?.slides ?? []).map(s => [s.idx, s.title])), [deck.data]);

  const seek = (t: number) => {
    if (!audio.current) return;
    audio.current.currentTime = Math.max(0, t - 0.3);
    void audio.current.play().catch(() => undefined);
  };

  const panels: TimelinePanel[] = useMemo(() => {
    const tl = r.sensors.timeline ?? [];
    const gaze = tl.filter(s => s.gaze !== null).map(s => ({ t: s.t, v: s.gaze === null ? null : s.gaze * 100 }));
    return [
      { key: 'wpm', label: 'Pace', unit: 'wpm', color: 'var(--viz-series-1)', domain: [60, 220], band: [130, 160], points: a.pace.windows.map(w => ({ t: (w.start + w.end) / 2, v: w.wpm || null })) },
      { key: 'pitch', label: 'Vocal variety', unit: 'semitones', color: 'var(--viz-series-2)', domain: [0, 14], band: [3, 14], points: a.prosody.windows.map(w => ({ t: w.t ?? w.start + 4, v: w.range_st })) },
      { key: 'gaze', label: 'Eye contact', unit: '%', color: 'var(--viz-series-3)', domain: [0, 100], band: [60, 100], points: gaze },
    ];
  }, [r, a]);

  const boundaries = r.segments.slice(1).map(s => ({ t: s.start, label: `S${s.slide_idx + 1}` }));
  const scoreItems = (Object.keys(METRIC_LABELS) as MetricKey[]).map(k => ({
    key: k,
    label: METRIC_LABELS[k],
    value: r.scores?.[k] ?? null,
    note: k === 'eye_contact' || k === 'posture' || k === 'composure' ? 'Needs the camera' : k === 'timing' ? 'Needs a target length' : undefined,
  }));
  const judge = r.judge;

  async function reanalyse() {
    setBusy(true);
    try {
      const updated = await rehearsalsApi.reanalyse(r.id);
      qc.setQueryData(['rehearsal', r.id], updated);
      toast('Report updated.');
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Re-analysis failed', 'error');
    } finally {
      setBusy(false);
    }
  }

  async function qa() {
    try {
      const { session_id } = await rehearsalsApi.startQa(r.id);
      navigate(`/coach?session=${session_id}`);
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Could not start Q&A', 'error');
    }
  }

  return (
    <div className={printMode ? 'report-print mx-auto max-w-4xl space-y-6 p-8' : 'h-full space-y-6 overflow-y-auto p-6'}>
      {/* Header */}
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs text-neutral-500">
            {new Date(r.created_at).toLocaleString()} · {fmt(r.duration_sec ?? 0)}
            {r.target_seconds ? ` of ${fmt(r.target_seconds)} target` : ''}
            {deck.data && (
              <>
                {' · '}
                <Link to={`/decks/${deck.data.id}`} className="underline">
                  {deck.data.title}
                </Link>
              </>
            )}
          </p>
          <h1 className="text-2xl font-bold">{r.title}</h1>
        </div>
        <div className="flex items-center gap-4">
          <ScoreTile label="Overall" value={r.overall_score} big testId="overall" />
          <ScoreTile label="Delivery" value={r.scores?.delivery} />
          <ScoreTile label="Content" value={r.scores?.content} />
        </div>
      </header>

      {!printMode && (
        <div className="flex flex-wrap gap-2 print:hidden">
          <button onClick={qa} className="flex items-center gap-1.5 rounded-xl bg-purple-600 px-3 py-2 text-sm font-semibold hover:bg-purple-500">
            <MessageCircleQuestion className="h-4 w-4" /> Practice investor Q&A on this
          </button>
          <Link to={`/history/compare?b=${r.id}`} className="flex items-center gap-1.5 rounded-xl border border-white/10 px-3 py-2 text-sm hover:bg-white/5">
            <ArrowLeftRight className="h-4 w-4" /> Compare
          </Link>
          <Link to={`/report/${r.id}/print`} className="flex items-center gap-1.5 rounded-xl border border-white/10 px-3 py-2 text-sm hover:bg-white/5">
            <Printer className="h-4 w-4" /> PDF report
          </Link>
          <button onClick={reanalyse} disabled={busy} className="flex items-center gap-1.5 rounded-xl border border-white/10 px-3 py-2 text-sm hover:bg-white/5 disabled:opacity-50">
            <RefreshCw className={`h-4 w-4 ${busy ? 'animate-spin' : ''}`} /> Re-analyse
          </button>
          {r.has_audio && (
            <button
              onClick={async () => {
                if (!window.confirm('Delete the audio recording? The report and transcript stay.')) return;
                await rehearsalsApi.deleteAudio(r.id);
                qc.invalidateQueries({ queryKey: ['rehearsal', r.id] });
                toast('Recording deleted.');
              }}
              className="ml-auto flex items-center gap-1.5 rounded-xl px-3 py-2 text-sm text-neutral-400 hover:bg-white/5"
            >
              Delete recording only
            </button>
          )}
          <button
            onClick={async () => {
              if (!window.confirm('Delete this rehearsal and its recording?')) return;
              await rehearsalsApi.remove(r.id);
              qc.invalidateQueries({ queryKey: ['rehearsals'] });
              navigate('/history');
            }}
            className={`${r.has_audio ? '' : 'ml-auto '}flex items-center gap-1.5 rounded-xl px-3 py-2 text-sm text-rose-300 hover:bg-rose-500/10`}
          >
            <Trash2 className="h-4 w-4" /> Delete
          </button>
        </div>
      )}
      {r.error && <p className="rounded-xl bg-amber-500/10 px-3 py-2 text-sm text-amber-200">{r.error}</p>}

      {/* Verdict & priorities */}
      {judge && (
        <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_300px]" data-testid="verdict">
          <div className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-neutral-400">Coach's verdict</h2>
            <p className="mt-2 text-neutral-200">{judge.summary}</p>
            <h3 className="mt-4 text-sm font-semibold">Top priorities for your next take</h3>
            <ol className="mt-2 space-y-3">
              {judge.priorities.map((p, i) => (
                <li key={i} className="rounded-xl bg-neutral-950/60 p-3">
                  <div className="font-semibold">
                    {i + 1}. {p.title}
                  </div>
                  <p className="text-sm text-neutral-300">{p.detail}</p>
                  {p.evidence.length > 0 && (
                    <div className="mt-1 flex flex-wrap gap-1">
                      {p.evidence.map((e, k) => {
                        const cue = r.cues.find(c => c.id === e);
                        return (
                          <button
                            key={k}
                            onClick={() => cue?.t != null && seek(cue.t)}
                            className="rounded-full bg-white/5 px-2 py-0.5 text-[11px] text-neutral-400 hover:text-white"
                            title={cue ? cue.message : 'Quote from your talk'}
                          >
                            {cue ? cue.message.slice(0, 48) + (cue.message.length > 48 ? '…' : '') : `“${e}”`}
                          </button>
                        );
                      })}
                    </div>
                  )}
                </li>
              ))}
            </ol>
            {judge.strengths.length > 0 && (
              <>
                <h3 className="mt-4 text-sm font-semibold">What worked</h3>
                <ul className="mt-1 list-disc pl-5 text-sm text-neutral-300">
                  {judge.strengths.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </>
            )}
          </div>
          <div className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-neutral-400">Pitch rubric</h2>
            <RadarChart title="6-dimension pitch rubric" data={DIMENSIONS.map(d => ({ label: d[0].toUpperCase() + d.slice(1), value: Math.round(judge.scorecard[d]?.score ?? 0) }))} />
            {judge.model && <p className="text-[10px] text-neutral-600">Judged by {judge.model.split(':').slice(1).join(':')}</p>}
          </div>
        </section>
      )}

      {/* Measured delivery */}
      <section className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
        <div className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-neutral-400">Delivery scores</h2>
          <ScoreBars items={scoreItems} />
          <dl className="mt-4 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
            <Fact k="Pace" v={`${Math.round(a.pace.wpm)} wpm`} />
            <Fact k="Fillers" v={`${a.fillers.count} (${a.fillers.per_minute}/min)`} />
            <Fact k="Pitch range" v={`${a.prosody.range_st} st${a.prosody.monotone ? ' · monotone' : ''}`} />
            <Fact k="Upspeak" v={`${a.upspeak.rising} of ${a.upspeak.sentences_checked}`} />
            <Fact k="Pauses" v={`${a.pauses.strategic} strategic · ${a.pauses.hesitation} hesitant`} />
            {r.sensors.camera && <Fact k="Eye contact" v={`${r.sensors.eye_contact_pct ?? '--'}%`} />}
            {r.sensors.camera && <Fact k="Posture" v={`${r.sensors.posture_avg ?? '--'}`} />}
          </dl>
        </div>
        <div className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4">
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-neutral-400">Over time</h2>
          <TimelineChart
            panels={panels}
            duration={r.duration_sec ?? a.duration}
            boundaries={boundaries}
            marks={a.fillers.items.map(f => ({ t: f.start, label: `Filler "${f.filler}"` }))}
            cursor={playhead}
            onSeek={seek}
          />
          <p className="mt-1 text-[11px] text-neutral-500">Shaded bands are the target range. Orange ticks along the bottom are fillers. Dashed lines are slide changes.</p>
          {r.has_audio && !printMode && (
            <audio
              ref={audio}
              controls
              src={rehearsalsApi.audioUrl(r.id)}
              className="mt-2 w-full"
              onTimeUpdate={e => setPlayhead((e.target as HTMLAudioElement).currentTime)}
            />
          )}
        </div>
      </section>

      {/* Coaching notes */}
      {r.cues.length > 0 && (
        <section className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4" data-testid="cues">
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-neutral-400">Coaching notes</h2>
          <ul className="grid gap-2 md:grid-cols-2">
            {r.cues.map(c => (
              <CueItem key={c.id} cue={c} onSeek={seek} />
            ))}
          </ul>
        </section>
      )}

      {/* Slides */}
      {r.segments.length > 1 || r.deck_id ? (
        <section className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4">
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-neutral-400">Slide by slide</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm" data-testid="slide-table">
              <thead className="text-left text-xs text-neutral-500">
                <tr>
                  <th className="py-1 pr-2">Slide</th>
                  <th className="pr-2">Time / budget</th>
                  <th className="pr-2">Pace</th>
                  <th className="pr-2">Fillers</th>
                  <th className="pr-2">Eye contact</th>
                  <th>Coach</th>
                </tr>
              </thead>
              <tbody>
                {r.segments.map((s, i) => {
                  const fb = judge?.slide_feedback.find(f => f.slide_idx === s.slide_idx);
                  const over = s.stats.budget && s.stats.duration > s.stats.budget * 1.25;
                  return (
                    <tr key={i} className="border-t border-white/5 align-top">
                      <td className="py-2 pr-2">
                        <button onClick={() => seek(s.start)} className="text-left hover:underline">
                          {s.slide_idx + 1}. {titles[s.slide_idx] ?? ''}
                        </button>
                      </td>
                      <td className={`pr-2 font-mono ${over ? 'text-rose-300' : ''}`}>
                        {fmt(s.stats.duration)}
                        {s.stats.budget ? ` / ${fmt(s.stats.budget)}` : ''}
                      </td>
                      <td className="pr-2 font-mono">{Math.round(s.stats.wpm)}</td>
                      <td className="pr-2 font-mono">{s.stats.fillers}</td>
                      <td className="pr-2 font-mono">{s.stats.eye_contact_pct == null ? '--' : `${Math.round(s.stats.eye_contact_pct)}%`}</td>
                      <td className="text-neutral-300">
                        {fb?.feedback}
                        {fb?.suggested_line && <div className="mt-1 text-purple-200">Try: “{fb.suggested_line}”</div>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {/* Transcript & language */}
      <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4">
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-neutral-400">Transcript</h2>
          <Transcript r={r} playhead={playhead} onSeek={seek} />
          <p className="mt-2 text-[11px] text-neutral-500">
            <mark className="bg-amber-500/20 text-amber-200">Fillers</mark>, <span className="underline decoration-sky-400 decoration-dotted">word-choice tips</span> and{' '}
            <span className="text-neutral-500 line-through decoration-rose-400/60">unclear words</span> are marked. Click a word to hear it.
          </p>
        </div>
        <div className="space-y-4">
          {a.language.items.length > 0 && (
            <div className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4">
              <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-neutral-400">Word choice</h2>
              <ul className="space-y-2 text-sm">
                {a.language.items.slice(0, 12).map((it, i) => (
                  <li key={i}>
                    <button onClick={() => seek(it.t)} className="font-semibold text-sky-300 hover:underline">
                      “{it.text}”
                    </button>{' '}
                    <span className="text-neutral-400">{it.tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {judge?.rewrite && (
            <div className="rounded-2xl border border-purple-500/30 bg-purple-950/20 p-4">
              <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-neutral-400">Rewrite your weakest moment</h2>
              <p className="text-sm text-neutral-500 line-through decoration-neutral-600">{judge.rewrite.original}</p>
              <p className="mt-2 text-sm text-purple-100">{judge.rewrite.improved}</p>
            </div>
          )}
          {(a.clarity.key_terms_missing.length > 0 || a.clarity.unclear_count > 0) && (
            <div className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4 text-sm">
              <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-neutral-400">Clarity</h2>
              <p className="text-neutral-300">Average recognition confidence {Math.round(a.clarity.avg_confidence * 100)}%.</p>
              {a.clarity.key_terms_missing.length > 0 && <p className="mt-1 text-neutral-400">Not mentioned from your slides: {a.clarity.key_terms_missing.slice(0, 6).join(', ')}</p>}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

function ScoreTile({ label, value, big, testId }: { label: string; value: number | null | undefined; big?: boolean; testId?: string }) {
  return (
    <div className="text-center">
      <div className={`font-black ${big ? 'text-4xl' : 'text-xl text-neutral-300'}`} data-testid={testId ? `score-${testId}` : undefined}>
        {value == null ? '--' : Math.round(value)}
      </div>
      <div className="text-[11px] uppercase tracking-wider text-neutral-500">{label}</div>
    </div>
  );
}

function Fact({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt className="text-neutral-500">{k}</dt>
      <dd className="text-right text-neutral-200">{v}</dd>
    </>
  );
}

const SEVERITY = {
  critical: { icon: XOctagon, cls: 'text-rose-300', label: 'Critical' },
  warn: { icon: AlertTriangle, cls: 'text-amber-300', label: 'Warning' },
  info: { icon: Info, cls: 'text-sky-300', label: 'Note' },
};

function CueItem({ cue, onSeek }: { cue: Cue; onSeek: (t: number) => void }) {
  const s = SEVERITY[cue.severity];
  const Icon = s.icon;
  return (
    <li className="flex gap-2 rounded-xl bg-neutral-950/60 p-2.5 text-sm">
      <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${s.cls}`} aria-label={s.label} />
      <span>
        {cue.message}
        {cue.t != null && (
          <button onClick={() => onSeek(cue.t!)} className="ml-1 font-mono text-xs text-neutral-500 hover:text-white">
            [{fmt(cue.t)}]
          </button>
        )}
      </span>
    </li>
  );
}

function Transcript({ r, playhead, onSeek }: { r: Rehearsal; playhead: number | null; onSeek: (t: number) => void }) {
  const a = r.analysis!;
  const fillerIdx = useMemo(() => {
    const set = new Set<number>();
    for (const f of a.fillers.items) {
      for (let i = f.word_index; i < r.words.length && r.words[i].start < f.end; i++) set.add(i);
    }
    return set;
  }, [a, r.words]);
  const tips = useMemo(() => new Map(a.language.items.map(i => [i.word_index, i.tip])), [a]);
  const unclear = useMemo(() => new Set(a.clarity.unclear.map(u => u.word_index)), [a]);
  const slideStarts = useMemo(() => new Map(r.segments.slice(1).map(s => [r.words.findIndex(w => w.start >= s.start), s.slide_idx])), [r]);

  return (
    <p className="max-h-96 overflow-y-auto text-sm leading-7" data-testid="transcript">
      {r.words.map((w, i) => {
        const active = playhead != null && playhead >= w.start && playhead < w.end + 0.1;
        const cls = [
          'cursor-pointer rounded hover:bg-white/10',
          fillerIdx.has(i) ? 'bg-amber-500/20 text-amber-200' : '',
          tips.has(i) ? 'underline decoration-sky-400 decoration-dotted' : '',
          unclear.has(i) ? 'text-neutral-500 line-through decoration-rose-400/60' : '',
          active ? 'bg-purple-500/40 text-white' : '',
        ].join(' ');
        return (
          <React.Fragment key={i}>
            {i > 0 && !attachesToPrevious(w.w) && ' '}
            {slideStarts.has(i) && <span className="mx-1 rounded bg-white/5 px-1 text-[10px] text-neutral-500">S{slideStarts.get(i)! + 1}</span>}
            <span className={cls} onClick={() => onSeek(w.start)} title={tips.get(i)}>
              {w.w.trim()}
            </span>
          </React.Fragment>
        );
      })}
    </p>
  );
}
