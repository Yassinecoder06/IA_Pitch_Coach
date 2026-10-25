import React, { useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { FileUp, Layers, ListOrdered, Loader2, Sparkles, Trash2 } from 'lucide-react';
import { decksApi, type DeckBrief } from '@/lib/decks';
import { importDeckFile, parseOutline } from './importDeck';
import { toast } from '@/stores/toast';

type Mode = null | 'outline' | 'upload' | 'ai';

const SAMPLE_OUTLINE = `Shipping is eating your margin [25s]
- Small retailers overpay 30% on freight
> Open with the pain: a store owner losing a third of their margin.

FreightPilot pools the volume [35s]
- One account, carrier rates of a big brand
- Books the cheapest carrier automatically

Raising $1.5M seed [20s]
- 40 paying stores today, 500 by next year`;

export function DecksPage() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const { data, isLoading } = useQuery({ queryKey: ['decks'], queryFn: decksApi.list });
  const [mode, setMode] = useState<Mode>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const remove = useMutation({
    mutationFn: (id: string) => decksApi.remove(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['decks'] }),
  });

  async function run(label: string, fn: () => Promise<{ id: string }>) {
    setBusy(label);
    try {
      const deck = await fn();
      await qc.invalidateQueries({ queryKey: ['decks'] });
      navigate(`/decks/${deck.id}`);
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Something went wrong', 'error');
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold">Decks</h1>
          <p className="text-sm text-neutral-400">Bring your slides, write an outline, or let the AI draft a pitch deck.</p>
        </div>
        <div className="flex gap-2">
          <ModeButton icon={ListOrdered} label="Write outline" active={mode === 'outline'} onClick={() => setMode(m => (m === 'outline' ? null : 'outline'))} />
          <ModeButton icon={FileUp} label="Upload PDF / PPTX" active={mode === 'upload'} onClick={() => setMode(m => (m === 'upload' ? null : 'upload'))} />
          <ModeButton icon={Sparkles} label="Generate with AI" active={mode === 'ai'} onClick={() => setMode(m => (m === 'ai' ? null : 'ai'))} />
        </div>
      </header>

      {busy && (
        <p className="mt-4 flex items-center gap-2 text-sm text-purple-300" role="status">
          <Loader2 className="h-4 w-4 animate-spin" /> {busy}
        </p>
      )}

      {mode === 'outline' && (
        <OutlineForm
          disabled={!!busy}
          onSubmit={(title, text) => run('Creating deck...', () => decksApi.create(title, 'outline', parseOutline(text)))}
        />
      )}
      {mode === 'upload' && <UploadForm disabled={!!busy} onFile={file => run('Importing...', () => importDeckFile(file, setBusy))} />}
      {mode === 'ai' && <GenerateForm disabled={!!busy} onSubmit={brief => run('Drafting your deck with AI...', () => decksApi.generate(brief))} />}

      <section className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {isLoading && <p className="text-sm text-neutral-500">Loading...</p>}
        {data?.decks.length === 0 && !mode && (
          <p className="col-span-full rounded-2xl border border-dashed border-white/10 p-8 text-center text-sm text-neutral-400">
            No decks yet. Create one above, then rehearse it slide by slide.
          </p>
        )}
        {data?.decks.map(d => (
          <article key={d.id} className="group rounded-2xl border border-white/10 bg-neutral-900/60 p-4 hover:border-purple-500/40">
            <Link to={`/decks/${d.id}`} className="block">
              <div className="flex items-center gap-2 text-xs text-neutral-500">
                <Layers className="h-3.5 w-3.5" /> {d.source.toUpperCase()} · {d.slide_count} slides
              </div>
              <h2 className="mt-1 font-semibold">{d.title}</h2>
              <p className="text-xs text-neutral-500">Updated {new Date(d.updated_at).toLocaleDateString()}</p>
            </Link>
            <div className="mt-3 flex items-center justify-between">
              <Link to={`/rehearse?deck=${d.id}`} className="rounded-lg bg-purple-600/80 px-3 py-1.5 text-xs font-semibold hover:bg-purple-500">
                Rehearse
              </Link>
              <button
                aria-label={`Delete ${d.title}`}
                onClick={() => window.confirm(`Delete "${d.title}"?`) && remove.mutate(d.id)}
                className="rounded-lg p-1.5 text-neutral-500 opacity-0 hover:bg-rose-500/10 hover:text-rose-300 group-hover:opacity-100 focus:opacity-100"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          </article>
        ))}
      </section>
    </div>
  );
}

function ModeButton({ icon: Icon, label, active, onClick }: { icon: React.ElementType; label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      className={`flex items-center gap-1.5 rounded-xl border px-3 py-2 text-sm ${
        active ? 'border-purple-500 bg-purple-600/20 text-white' : 'border-white/10 text-neutral-300 hover:bg-white/5'
      }`}
    >
      <Icon className="h-4 w-4" /> {label}
    </button>
  );
}

const field = 'w-full rounded-xl border border-white/10 bg-neutral-900 px-3 py-2 text-sm focus:border-purple-500 focus:outline-none';

function OutlineForm({ disabled, onSubmit }: { disabled: boolean; onSubmit: (title: string, text: string) => void }) {
  const [title, setTitle] = useState('');
  const [text, setText] = useState(SAMPLE_OUTLINE);
  const slides = parseOutline(text);
  return (
    <form
      className="mt-4 space-y-3 rounded-2xl border border-white/10 bg-neutral-900/50 p-4"
      onSubmit={e => {
        e.preventDefault();
        onSubmit(title.trim() || 'My pitch', text);
      }}
    >
      <input className={field} value={title} onChange={e => setTitle(e.target.value)} placeholder="Deck title" aria-label="Deck title" />
      <textarea className={`${field} h-56 font-mono`} value={text} onChange={e => setText(e.target.value)} aria-label="Outline" />
      <p className="text-xs text-neutral-500">
        One slide per paragraph. First line is the title (add <code>[30s]</code> for a time budget), <code>- </code> lines are bullets,{' '}
        <code>&gt; </code> lines are speaker notes. {slides.length} slide{slides.length === 1 ? '' : 's'}.
      </p>
      <button disabled={disabled || !slides.length} className="rounded-xl bg-purple-600 px-4 py-2 text-sm font-semibold hover:bg-purple-500 disabled:opacity-50">
        Create deck
      </button>
    </form>
  );
}

function UploadForm({ disabled, onFile }: { disabled: boolean; onFile: (f: File) => void }) {
  const input = useRef<HTMLInputElement | null>(null);
  const [drag, setDrag] = useState(false);
  return (
    <div
      onDragOver={e => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => {
        e.preventDefault();
        setDrag(false);
        const f = e.dataTransfer.files[0];
        if (f) onFile(f);
      }}
      className={`mt-4 rounded-2xl border-2 border-dashed p-8 text-center ${drag ? 'border-purple-500 bg-purple-500/5' : 'border-white/10'}`}
    >
      <FileUp className="mx-auto h-8 w-8 text-neutral-500" />
      <p className="mt-2 text-sm">Drop a PDF or PowerPoint here</p>
      <p className="text-xs text-neutral-500">Pages are rendered on your computer; PPTX keeps its speaker notes.</p>
      <input
        ref={input}
        type="file"
        accept=".pdf,.pptx,application/pdf,application/vnd.openxmlformats-officedocument.presentationml.presentation"
        className="hidden"
        data-testid="deck-file"
        onChange={e => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = '';
        }}
      />
      <button disabled={disabled} onClick={() => input.current?.click()} className="mt-3 rounded-xl bg-purple-600 px-4 py-2 text-sm font-semibold hover:bg-purple-500 disabled:opacity-50">
        Choose file
      </button>
    </div>
  );
}

function GenerateForm({ disabled, onSubmit }: { disabled: boolean; onSubmit: (b: DeckBrief) => void }) {
  const [brief, setBrief] = useState<DeckBrief>({ company: '', one_liner: '', audience: 'seed investors', stage: 'pre-seed', minutes: 3, details: '' });
  const set = (k: keyof DeckBrief) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) =>
    setBrief(b => ({ ...b, [k]: k === 'minutes' ? Number(e.target.value) : e.target.value }));
  return (
    <form
      className="mt-4 grid gap-3 rounded-2xl border border-white/10 bg-neutral-900/50 p-4 sm:grid-cols-2"
      onSubmit={e => {
        e.preventDefault();
        onSubmit(brief);
      }}
    >
      <input required className={field} placeholder="Company name" aria-label="Company name" value={brief.company} onChange={set('company')} />
      <input required minLength={3} className={field} placeholder="What it does, in one sentence" aria-label="One-liner" value={brief.one_liner} onChange={set('one_liner')} />
      <input className={field} placeholder="Audience" aria-label="Audience" value={brief.audience} onChange={set('audience')} />
      <div className="flex gap-2">
        <select className={field} value={brief.stage} onChange={set('stage')} aria-label="Stage">
          {['idea', 'pre-seed', 'seed', 'series A', 'growth'].map(s => (
            <option key={s}>{s}</option>
          ))}
        </select>
        <select className={field} value={brief.minutes} onChange={set('minutes')} aria-label="Length">
          {[1, 2, 3, 5, 7, 10].map(m => (
            <option key={m} value={m}>
              {m} min
            </option>
          ))}
        </select>
      </div>
      <textarea
        className={`${field} h-28 sm:col-span-2`}
        placeholder="Facts the deck can use: customers, metrics, team, how much you are raising..."
        aria-label="Details"
        value={brief.details}
        onChange={set('details')}
      />
      <p className="text-xs text-neutral-500 sm:col-span-2">Uses your "Deck generation" model from Settings. Unknown numbers become placeholders like [X customers].</p>
      <button disabled={disabled} className="w-fit rounded-xl bg-purple-600 px-4 py-2 text-sm font-semibold hover:bg-purple-500 disabled:opacity-50">
        Generate deck
      </button>
    </form>
  );
}
