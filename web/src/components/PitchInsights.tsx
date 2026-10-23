import React from 'react';
import {
  CheckCircle2,
  XCircle,
  Sparkles,
  Gauge,
  TrendingUp,
  AlertCircle,
  Lightbulb,
  Award,
} from 'lucide-react';
import { ScoreRadar } from './ScoreRadar';
import type { Scorecard, SpeechMetrics } from '../lib/types';

interface PitchInsightsProps {
  scorecard: Scorecard | null;
  speechMetrics: SpeechMetrics | null;
}

export function PitchInsights({ scorecard, speechMetrics }: PitchInsightsProps) {
  if (!scorecard && !speechMetrics) {
    return (
      <aside className="glass-panel hidden lg:flex w-80 xl:w-96 flex-col justify-center items-center p-8 text-center border-l border-white/5">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-purple-500/10 text-purple-400 mb-4">
          <Award className="h-7 w-7" />
        </div>
        <h3 className="text-base font-bold text-white">Live SOTA Insights</h3>
        <p className="mt-2 text-xs text-neutral-400 leading-relaxed max-w-xs">
          Record or send a pitch turn. Real-time radar metrics, structure detection, and pacing analysis will appear here.
        </p>
      </aside>
    );
  }

  const overall = scorecard?.overall ?? 75;
  const checklist = scorecard?.checklist || {
    hook: false,
    problem: false,
    solution: false,
    market: false,
    traction: false,
    ask: false,
  };

  const wpm = speechMetrics?.words_per_minute || 0;
  const fillers = speechMetrics?.filler_count || 0;
  const pauseFreq = speechMetrics?.pause_frequency || 0;

  // Pace indicator
  const paceStatus =
    wpm === 0
      ? 'No speech'
      : wpm < 120
      ? 'Slow cadence'
      : wpm > 170
      ? 'Fast cadence'
      : 'Optimal pacing (130-160 WPM)';

  return (
    <aside className="glass-panel w-full lg:w-80 xl:w-96 flex-col border-l border-white/5 overflow-y-auto p-5 space-y-6">
      {/* Top Banner: Overall Score Ring */}
      <div className="glass-card flex items-center justify-between rounded-2xl p-4 border border-purple-500/20">
        <div>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">
            Pitch Quality Score
          </span>
          <h2 className="text-2xl font-black text-white mt-0.5">
            {overall}
            <span className="text-sm font-normal text-neutral-400">/100</span>
          </h2>
          <span className="text-xs text-purple-300">
            {overall >= 85 ? 'Investor Ready 🚀' : overall >= 70 ? 'Solid Foundation 👍' : 'Needs Practice 💪'}
          </span>
        </div>

        {/* Circular Progress Indicator */}
        <div className="relative flex h-16 w-16 items-center justify-center">
          <svg className="h-full w-full -rotate-90" viewBox="0 0 36 36">
            <path
              className="text-white/10"
              strokeWidth="3.5"
              stroke="currentColor"
              fill="none"
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            />
            <path
              className="text-purple-500 transition-all duration-1000 ease-out"
              strokeDasharray={`${overall}, 100`}
              strokeLinecap="round"
              strokeWidth="3.5"
              stroke="currentColor"
              fill="none"
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            />
          </svg>
          <span className="absolute text-xs font-bold text-white">{overall}%</span>
        </div>
      </div>

      {/* 6D Radar Visualizer */}
      {scorecard && (
        <div className="glass-card rounded-2xl p-4">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 mb-2">
            Multi-Dimensional Radar
          </h4>
          <ScoreRadar scorecard={scorecard} size={250} />
        </div>
      )}

      {/* Pitch Structure Checklist */}
      <div className="glass-card rounded-2xl p-4">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 mb-3">
          Essential Pitch Structure
        </h4>
        <div className="grid grid-cols-2 gap-2 text-xs">
          {[
            { key: 'hook', label: 'Hook / Opener' },
            { key: 'problem', label: 'Problem & Pain' },
            { key: 'solution', label: 'Product Solution' },
            { key: 'market', label: 'Market Opportunity' },
            { key: 'traction', label: 'Traction / Metric' },
            { key: 'ask', label: 'Clear Funding Ask' },
          ].map(item => {
            const hasIt = Boolean(checklist[item.key as keyof typeof checklist]);
            return (
              <div
                key={item.key}
                className={`flex items-center gap-1.5 rounded-xl px-2.5 py-2 border ${
                  hasIt
                    ? 'border-emerald-500/20 bg-emerald-500/5 text-emerald-300'
                    : 'border-white/5 bg-neutral-900/40 text-neutral-500'
                }`}
              >
                {hasIt ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                ) : (
                  <XCircle className="h-3.5 w-3.5 text-neutral-500 shrink-0" />
                )}
                <span className="truncate">{item.label}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Speech Delivery & Rhythm Gauges */}
      <div className="glass-card rounded-2xl p-4 space-y-3">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
          Delivery Metrics
        </h4>

        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-neutral-300">Speaking Pace</span>
            <span className="font-semibold text-purple-300">{wpm ? `${wpm.toFixed(0)} WPM` : '--'}</span>
          </div>
          <div className="h-2 w-full rounded-full bg-white/5 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                wpm >= 130 && wpm <= 165 ? 'bg-emerald-500' : 'bg-purple-500'
              }`}
              style={{ width: `${Math.min(Math.max((wpm / 200) * 100, 5), 100)}%` }}
            />
          </div>
          <span className="text-[11px] text-neutral-500 mt-1 block">{paceStatus}</span>
        </div>

        <div className="grid grid-cols-2 gap-2 pt-2 border-t border-white/5">
          <div className="rounded-xl bg-neutral-900/50 p-2.5">
            <span className="text-[11px] text-neutral-400 block">Filler Words</span>
            <span className={`text-base font-bold ${fillers > 4 ? 'text-amber-400' : 'text-white'}`}>
              {fillers}
            </span>
          </div>
          <div className="rounded-xl bg-neutral-900/50 p-2.5">
            <span className="text-[11px] text-neutral-400 block">Pauses / Min</span>
            <span className="text-base font-bold text-white">
              {pauseFreq ? pauseFreq.toFixed(1) : '--'}
            </span>
          </div>
        </div>
      </div>

      {/* Top Actionable Fixes */}
      {scorecard?.top_fixes && scorecard.top_fixes.length > 0 && (
        <div className="glass-card rounded-2xl p-4 border border-amber-500/20">
          <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-amber-400 mb-2">
            <Lightbulb className="h-3.5 w-3.5" /> High-Impact Improvements
          </div>
          <ul className="space-y-2 text-xs text-neutral-300">
            {scorecard.top_fixes.map((fix, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-amber-500/20 text-[10px] font-bold text-amber-300">
                  {idx + 1}
                </span>
                <span>{fix}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Rewrite Suggestion */}
      {scorecard?.rewrite && (
        <div className="glass-card rounded-2xl p-4 border border-purple-500/20">
          <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-purple-400 mb-2">
            <Sparkles className="h-3.5 w-3.5" /> Suggested Polish
          </div>
          <p className="text-xs text-neutral-300 italic leading-relaxed bg-neutral-900/60 p-3 rounded-xl border border-white/5">
            "{scorecard.rewrite}"
          </p>
        </div>
      )}
    </aside>
  );
}
