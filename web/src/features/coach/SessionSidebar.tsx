import React, { useState } from 'react';
import {
  Plus,
  Search,
  MessageSquare,
  Trash2,
  Edit2,
  Sparkles,
  Check,
  X,
} from 'lucide-react';
import type { Session } from '@/lib/types';

interface SessionSidebarProps {
  sessions: Session[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onRenameSession: (id: string, newTitle: string) => void;
  onDeleteSession: (id: string) => void;
}

export function SessionSidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onRenameSession,
  onDeleteSession,
}: SessionSidebarProps) {
  const [search, setSearch] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  const filtered = sessions.filter(s =>
    s.title.toLowerCase().includes(search.toLowerCase())
  );

  const startRename = (s: Session, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(s.id);
    setEditTitle(s.title);
  };

  const saveRename = (id: string, e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (editTitle.trim()) {
      onRenameSession(id, editTitle.trim());
    }
    setEditingId(null);
  };

  return (
    <>
      <aside className="glass-panel flex h-full w-72 xl:w-80 flex-col border-r border-white/5 bg-[#0e0d16]/90 p-4 shrink-0 select-none">
        {/* Brand Header */}
        <div className="flex items-center justify-between pb-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-500 shadow-md shadow-purple-500/20">
              <Sparkles className="h-4 w-4 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-tight text-white">AI Pitch Coach</h1>
              <span className="text-[10px] text-purple-400 font-mono">Local rehearsal studio</span>
            </div>
          </div>
        </div>

        {/* New Session Button */}
        <button
          onClick={onNewSession}
          className="mt-2 flex w-full items-center justify-center gap-2 rounded-xl bg-purple-600/90 py-2.5 text-xs font-semibold text-white shadow-lg shadow-purple-600/20 transition-all hover:bg-purple-500 hover:scale-[1.01]"
        >
          <Plus className="h-4 w-4" /> New Pitch Session
        </button>

        {/* Search */}
        <div className="relative mt-4">
          <Search className="pointer-events-none absolute left-3 top-2.5 h-3.5 w-3.5 text-neutral-500" />
          <input
            type="search"
            placeholder="Search sessions..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="w-full rounded-xl border border-white/5 bg-neutral-900/60 py-2 pl-8 pr-3 text-xs text-white placeholder:text-neutral-600 focus:border-purple-500 focus:outline-none"
          />
        </div>

        {/* Session List */}
        <div className="mt-4 flex-1 overflow-y-auto space-y-1 pr-1">
          {filtered.map(s => {
            const isActive = s.id === activeSessionId;
            const isEditing = s.id === editingId;

            return (
              <div
                key={s.id}
                onClick={() => onSelectSession(s.id)}
                className={`group relative flex items-center justify-between rounded-xl px-3 py-2.5 text-xs transition-all cursor-pointer ${
                  isActive
                    ? 'bg-purple-600/20 border border-purple-500/30 text-white font-medium'
                    : 'text-neutral-400 hover:bg-white/5 hover:text-neutral-200 border border-transparent'
                }`}
              >
                {isEditing ? (
                  <form
                    onSubmit={e => saveRename(s.id, e)}
                    className="flex flex-1 items-center gap-1.5"
                    onClick={e => e.stopPropagation()}
                  >
                    <input
                      type="text"
                      value={editTitle}
                      onChange={e => setEditTitle(e.target.value)}
                      className="w-full rounded-lg bg-neutral-900 px-2 py-1 text-xs text-white focus:outline-none focus:ring-1 focus:ring-purple-500"
                      autoFocus
                    />
                    <button type="submit" className="text-emerald-400 hover:text-emerald-300">
                      <Check className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => setEditingId(null)}
                      className="text-neutral-500 hover:text-neutral-300"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </form>
                ) : (
                  <>
                    <div className="flex items-center gap-2.5 truncate pr-2">
                      <MessageSquare className={`h-3.5 w-3.5 shrink-0 ${isActive ? 'text-purple-400' : 'text-neutral-500'}`} />
                      <span className="truncate">{s.title}</span>
                    </div>

                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={e => startRename(s, e)}
                        className="rounded p-1 text-neutral-400 hover:text-white"
                        title="Rename"
                      >
                        <Edit2 className="h-3 w-3" />
                      </button>
                      <button
                        onClick={e => {
                          e.stopPropagation();
                          setConfirmDeleteId(s.id);
                        }}
                        className="rounded p-1 text-neutral-400 hover:text-rose-400"
                        title="Delete"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  </>
                )}
              </div>
            );
          })}

          {filtered.length === 0 && (
            <p className="p-4 text-center text-xs text-neutral-500">
              {search ? 'No sessions match your search' : 'No saved sessions yet'}
            </p>
          )}
        </div>

      </aside>

      {/* Delete Confirmation Modal (Matching dark UI popup, no native alert) */}
      {confirmDeleteId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="glass-panel w-full max-w-sm rounded-2xl p-6 shadow-2xl border border-white/10 text-center">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-rose-500/10 text-rose-400 mb-3">
              <Trash2 className="h-6 w-6" />
            </div>
            <h3 className="text-base font-bold text-white">Delete Session?</h3>
            <p className="mt-1.5 text-xs text-neutral-400">
              This will permanently delete this pitch session and all associated turns and metrics.
            </p>

            <div className="mt-6 flex items-center justify-center gap-2.5">
              <button
                type="button"
                onClick={() => setConfirmDeleteId(null)}
                className="rounded-xl border border-white/10 px-4 py-2 text-xs font-semibold text-neutral-300 hover:bg-white/5"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  onDeleteSession(confirmDeleteId);
                  setConfirmDeleteId(null);
                }}
                className="rounded-xl bg-rose-600 px-4 py-2 text-xs font-semibold text-white shadow-lg shadow-rose-600/20 hover:bg-rose-500"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
