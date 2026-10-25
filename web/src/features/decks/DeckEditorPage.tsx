import React, { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ArrowDown, ArrowUp, Download, Loader2, MessageSquareText, Play, Plus, Save, Trash2 } from 'lucide-react';
import { decksApi, type Deck, type Slide } from '@/lib/decks';
import { SlideView } from './SlideView';
import { toast } from '@/stores/toast';

type Draft = Omit<Slide, 'idx'>;

const toDraft = (deck: Deck): Draft[] => deck.slides.map(({ idx: _idx, ...rest }) => rest);
const field = 'w-full rounded-lg border border-white/10 bg-neutral-900 px-2 py-1.5 text-sm focus:border-purple-500 focus:outline-none';

export function DeckEditorPage() {
  const { deckId = '' } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const { data: deck, isLoading, error } = useQuery({ queryKey: ['deck', deckId], queryFn: () => decksApi.get(deckId) });
  const [title, setTitle] = useState('');
  const [slides, setSlides] = useState<Draft[]>([]);
  const [selected, setSelected] = useState(0);
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    if (deck) {
      setTitle(deck.title);
      setSlides(toDraft(deck));
      setDirty(false);
    }
  }, [deck]);

  const current = slides[selected];
  const totalBudget = useMemo(() => slides.reduce((s, x) => s + (x.time_budget_sec ?? 0), 0), [slides]);
  const critique = deck?.critique;

  if (isLoading) return <p className="p-6 text-sm text-neutral-500">Loading deck...</p>;
  if (error || !deck) return <p className="p-6 text-sm text-rose-400">Deck not found.</p>;

  function patch(i: number, change: Partial<Draft>) {
    setSlides(list => list.map((s, k) => (k === i ? { ...s, ...change } : s)));
    setDirty(true);
  }

  function move(i: number, dir: -1 | 1) {
    const j = i + dir;
    if (j < 0 || j >= slides.length || slides.some(s => s.has_image)) return;
    setSlides(list => {
      const copy = [...list];
      [copy[i], copy[j]] = [copy[j], copy[i]];
      return copy;
    });
    setSelected(j);
    setDirty(true);
  }

  async function save() {
    setBusy('Saving...');
    try {
      if (title.trim() && title !== deck!.title) await decksApi.rename(deckId, title.trim());
      const clean = slides.map(s => ({ ...s, bullets: s.bullets.map(b => b.trim()).filter(Boolean) }));
      const updated = await decksApi.replaceSlides(deckId, clean);
      qc.setQueryData(['deck', deckId], updated);
      qc.invalidateQueries({ queryKey: ['decks'] });
      setDirty(false);
      toast('Deck saved.');
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Save failed', 'error');
    } finally {
      setBusy(null);
    }
  }

  async function runCritique() {
    if (dirty) await save();
    setBusy('Reviewing your slides...');
    try {
      await decksApi.critique(deckId);
      await qc.invalidateQueries({ queryKey: ['deck', deckId] });
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Critique failed', 'error');
    } finally {
      setBusy(null);
    }
  }

  const imageDeck = slides.some(s => s.has_image);
  const slideCritique = critique?.slides.find(c => c.slide_number === selected + 1);

  return (
    <div className="grid h-full grid-cols-1 overflow-hidden lg:grid-cols-[240px_minmax(0,1fr)_320px]">
      {/* Slide rail */}
      <aside className="overflow-y-auto border-r border-white/5 p-3" aria-label="Slides">
        <Link to="/decks" className="text-xs text-neutral-500 hover:text-white">
          ← All decks
        </Link>
        <ol className="mt-3 space-y-2">
          {slides.map((s, i) => (
            <li key={i}>
              <button
                onClick={() => setSelected(i)}
                aria-current={i === selected}
                className={`w-full rounded-lg p-1 text-left ${i === selected ? 'ring-2 ring-purple-500' : 'hover:bg-white/5'}`}
              >
                <SlideView deckId={deckId} slide={{ ...s, idx: i }} compact />
                <span className="mt-1 block truncate px-1 text-[11px] text-neutral-400">
                  {i + 1}. {s.title || 'Untitled'}
                </span>
              </button>
            </li>
          ))}
        </ol>
        {!imageDeck && (
          <button
            onClick={() => {
              setSlides(list => [...list, { kind: null, title: 'New slide', bullets: [], notes: '', text: '', has_image: false, time_budget_sec: 20 }]);
              setSelected(slides.length);
              setDirty(true);
            }}
            className="mt-3 flex w-full items-center justify-center gap-1 rounded-lg border border-dashed border-white/15 py-2 text-xs text-neutral-400 hover:text-white"
          >
            <Plus className="h-3.5 w-3.5" /> Add slide
          </button>
        )}
      </aside>

      {/* Editor */}
      <section className="flex min-h-0 flex-col gap-3 overflow-y-auto p-4">
        <div className="flex flex-wrap items-center gap-2">
          <input
            className="min-w-[200px] flex-1 rounded-xl border border-white/10 bg-transparent px-3 py-2 text-lg font-bold focus:border-purple-500 focus:outline-none"
            value={title}
            onChange={e => {
              setTitle(e.target.value);
              setDirty(true);
            }}
            aria-label="Deck title"
          />
          <button onClick={save} disabled={!dirty || !!busy} className="flex items-center gap-1.5 rounded-xl border border-white/10 px-3 py-2 text-sm hover:bg-white/5 disabled:opacity-40">
            <Save className="h-4 w-4" /> Save
          </button>
          <a href={decksApi.exportUrl(deckId)} className="flex items-center gap-1.5 rounded-xl border border-white/10 px-3 py-2 text-sm hover:bg-white/5">
            <Download className="h-4 w-4" /> PPTX
          </a>
          <button
            onClick={async () => {
              if (dirty) await save();
              navigate(`/rehearse?deck=${deckId}`);
            }}
            className="flex items-center gap-1.5 rounded-xl bg-purple-600 px-3 py-2 text-sm font-semibold hover:bg-purple-500"
          >
            <Play className="h-4 w-4" /> Rehearse this deck
          </button>
        </div>
        {busy && (
          <p className="flex items-center gap-2 text-sm text-purple-300" role="status">
            <Loader2 className="h-4 w-4 animate-spin" /> {busy}
          </p>
        )}

        {current && (
          <>
            <div className="mx-auto w-full max-w-3xl">
              <SlideView deckId={deckId} slide={{ ...current, idx: selected }} version={deck.updated_at} />
            </div>
            <div className="mx-auto grid w-full max-w-3xl gap-2">
              <label className="text-xs text-neutral-400">
                Title
                <input className={field} value={current.title} onChange={e => patch(selected, { title: e.target.value })} />
              </label>
              {!current.has_image && (
                <label className="text-xs text-neutral-400">
                  Bullets (one per line)
                  <textarea
                    className={`${field} h-24`}
                    value={current.bullets.join('\n')}
                    onChange={e => patch(selected, { bullets: e.target.value.split('\n') })}
                  />
                </label>
              )}
              <label className="text-xs text-neutral-400">
                Speaker notes (shown on the teleprompter)
                <textarea className={`${field} h-28`} value={current.notes} onChange={e => patch(selected, { notes: e.target.value })} />
              </label>
              <div className="flex flex-wrap items-end gap-2">
                <label className="text-xs text-neutral-400">
                  Time budget (s)
                  <input
                    type="number"
                    min={1}
                    className={`${field} w-28`}
                    value={current.time_budget_sec ?? ''}
                    onChange={e => patch(selected, { time_budget_sec: e.target.value ? Number(e.target.value) : null })}
                  />
                </label>
                <span className="pb-2 text-xs text-neutral-500">Deck total: {Math.round(totalBudget / 6) / 10} min</span>
                <div className="ml-auto flex gap-1 pb-1">
                  {!imageDeck && (
                    <>
                      <button aria-label="Move slide up" onClick={() => move(selected, -1)} className="rounded-lg p-2 hover:bg-white/5"><ArrowUp className="h-4 w-4" /></button>
                      <button aria-label="Move slide down" onClick={() => move(selected, 1)} className="rounded-lg p-2 hover:bg-white/5"><ArrowDown className="h-4 w-4" /></button>
                      <button
                        aria-label="Delete slide"
                        disabled={slides.length <= 1}
                        onClick={() => {
                          setSlides(list => list.filter((_, k) => k !== selected));
                          setSelected(s => Math.max(0, s - 1));
                          setDirty(true);
                        }}
                        className="rounded-lg p-2 text-rose-300 hover:bg-rose-500/10 disabled:opacity-30"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          </>
        )}
      </section>

      {/* Critique */}
      <aside className="overflow-y-auto border-l border-white/5 p-4">
        <button onClick={runCritique} disabled={!!busy} className="flex w-full items-center justify-center gap-1.5 rounded-xl border border-purple-500/40 px-3 py-2 text-sm text-purple-200 hover:bg-purple-500/10 disabled:opacity-50">
          <MessageSquareText className="h-4 w-4" /> {critique ? 'Re-run slide critique' : 'Critique my slides'}
        </button>
        {critique ? (
          <div className="mt-4 space-y-3 text-sm" data-testid="critique">
            <p className="text-neutral-300">{critique.overall}</p>
            {critique.missing_sections.length > 0 && (
              <p className="text-xs text-amber-300">Missing: {critique.missing_sections.join(', ')}</p>
            )}
            <div className="rounded-xl bg-neutral-900/70 p-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">Slide {selected + 1}</h3>
              {slideCritique ? (
                <>
                  <span className={`mt-1 inline-block rounded-full px-2 py-0.5 text-[10px] ${slideCritique.density === 'ok' ? 'bg-emerald-500/15 text-emerald-300' : 'bg-amber-500/15 text-amber-300'}`}>
                    {slideCritique.density.replace('_', ' ')}
                  </span>
                  <ul className="mt-2 list-disc space-y-1 pl-4 text-neutral-300">
                    {slideCritique.issues.map((i, k) => <li key={k}>{i}</li>)}
                  </ul>
                  {slideCritique.suggestion && <p className="mt-2 text-purple-200">{slideCritique.suggestion}</p>}
                </>
              ) : (
                <p className="mt-1 text-neutral-500">No issues flagged.</p>
              )}
              <p className="mt-2 text-[11px] text-neutral-500">
                {critique.density.find(d => d.slide_number === selected + 1)?.words ?? 0} words on this slide
              </p>
            </div>
          </div>
        ) : (
          <p className="mt-4 text-xs text-neutral-500">Get feedback on density, one idea per slide, and missing pitch sections.</p>
        )}
      </aside>
    </div>
  );
}
