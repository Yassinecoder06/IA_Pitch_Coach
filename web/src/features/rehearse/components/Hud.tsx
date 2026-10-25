import React from 'react';
import { AlertTriangle, Eye, Gauge, MessageSquareWarning, PersonStanding, Waves } from 'lucide-react';
import { liveWpm, useSensors } from '@/stores/sensors';
import type { PostureIssue } from '@/lib/sensors/vision/posture';

export const ISSUE_TEXT: Record<PostureIssue, string> = {
  slouching: 'Slouching',
  shoulders_tilted: 'Shoulders tilted',
  arms_crossed: 'Arms crossed',
  hand_on_face: 'Hand on face',
  hands_hidden: 'Hands hidden',
};

function Card({ icon: Icon, label, children, tone = 'neutral', testId }: {
  icon: React.ElementType;
  label: string;
  children: React.ReactNode;
  tone?: 'good' | 'warn' | 'bad' | 'neutral';
  testId?: string;
}) {
  const ring = { good: 'border-emerald-500/40', warn: 'border-amber-500/40', bad: 'border-rose-500/40', neutral: 'border-white/10' }[tone];
  return (
    <div className={`rounded-2xl border bg-neutral-900/70 p-3 ${ring}`} data-testid={testId}>
      <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-neutral-400">
        <Icon className="h-3.5 w-3.5" /> {label}
      </div>
      <div className="mt-1">{children}</div>
    </div>
  );
}

function Sparkline({ values }: { values: number[] }) {
  const voiced = values.filter(v => v > 0);
  if (voiced.length < 2) return <div className="h-8" />;
  const lo = Math.min(...voiced);
  const hi = Math.max(...voiced, lo + 1);
  const pts = values
    .map((v, i) => (v > 0 ? `${(i / (values.length - 1)) * 100},${30 - ((v - lo) / (hi - lo)) * 28}` : null))
    .filter(Boolean)
    .join(' ');
  return (
    <svg viewBox="0 0 100 32" className="h-8 w-full" preserveAspectRatio="none" aria-hidden>
      <polyline points={pts} fill="none" stroke="currentColor" strokeWidth="1.5" className="text-purple-400" />
    </svg>
  );
}

export function Hud() {
  const { voice, gaze, posture, motion, vision, words, fillers } = useSensors();
  const wpm = liveWpm(words);
  const visionOn = vision?.state === 'running';

  const gazeTone = !visionOn ? 'neutral' : gaze?.state === 'good' ? 'good' : gaze?.state === 'weak' ? 'warn' : 'bad';
  const paceTone = wpm === 0 ? 'neutral' : wpm < 110 || wpm > 175 ? 'bad' : wpm < 130 || wpm > 160 ? 'warn' : 'good';
  const postureTone = !posture?.bodyPresent ? 'neutral' : posture.score >= 80 ? 'good' : posture.score >= 55 ? 'warn' : 'bad';

  return (
    <div className="grid grid-cols-2 gap-2 xl:grid-cols-3" aria-live="off">
      <Card icon={Eye} label="Eye contact" tone={gazeTone} testId="hud-eye">
        <div className="text-xl font-bold">
          {!visionOn ? '--' : !gaze?.facePresent ? 'No face' : gaze.state === 'good' ? 'Good' : gaze.state === 'weak' ? 'Drifting' : 'Lost'}
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
          <div className="h-full bg-emerald-400 transition-all" style={{ width: `${Math.round((gaze?.score ?? 0) * 100)}%` }} />
        </div>
      </Card>

      <Card icon={PersonStanding} label="Posture" tone={postureTone} testId="hud-posture">
        <div className="text-xl font-bold">{posture?.bodyPresent ? posture.score : '--'}</div>
        <div className="flex flex-wrap gap-1">
          {(posture?.issues ?? []).map(i => (
            <span key={i} className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] text-amber-300">
              {ISSUE_TEXT[i]}
            </span>
          ))}
          {motion && motion.fidget > 0.5 && <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] text-amber-300">Fidgeting</span>}
          {motion && motion.sway > 0.5 && <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] text-amber-300">Swaying</span>}
        </div>
      </Card>

      <Card icon={Waves} label="Vocal variety" tone={voice?.monotone ? 'bad' : voice?.pitchRangeSt ? 'good' : 'neutral'} testId="hud-pitch">
        <div className="flex items-baseline gap-2">
          <span className="text-xl font-bold">{voice?.pitchRangeSt ? `${voice.pitchRangeSt.toFixed(1)} st` : '--'}</span>
          {voice?.monotone && <span className="text-xs font-semibold text-rose-300">Monotone</span>}
          {voice && voice.f0 > 0 && <span className="text-[11px] text-neutral-500">{Math.round(voice.f0)} Hz</span>}
        </div>
        <Sparkline values={voice?.pitchTrail ?? []} />
      </Card>

      <Card icon={Gauge} label="Pace" tone={paceTone} testId="hud-pace">
        <div className="text-xl font-bold">{wpm ? `${wpm} wpm` : '--'}</div>
        <div className="text-[11px] text-neutral-500">Target 130-160</div>
      </Card>

      <Card icon={MessageSquareWarning} label="Fillers" tone={fillers.length > 5 ? 'warn' : 'neutral'} testId="hud-fillers">
        <div className="text-xl font-bold">{fillers.length}</div>
        <div className="truncate text-[11px] text-neutral-500">{fillers.slice(-4).map(f => f.filler).join(' · ') || 'None yet'}</div>
      </Card>

      <Card icon={AlertTriangle} label="Energy & pauses" testId="hud-energy">
        <div className="text-sm">
          <span className="font-bold">{voice?.energyRangeDb ? `${voice.energyRangeDb.toFixed(0)} dB` : '--'}</span>
          <span className="text-neutral-500"> range · </span>
          <span className="font-bold">{voice?.pauseCount ?? 0}</span>
          <span className="text-neutral-500"> pauses</span>
        </div>
        <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
          <div className="h-full bg-purple-400 transition-all" style={{ width: `${Math.max(0, Math.min(100, ((voice?.levelDb ?? -60) + 60) * 2))}%` }} />
        </div>
      </Card>

      {vision && (
        <p className="col-span-full text-[10px] text-neutral-600" data-testid="vision-status">
          Vision: {vision.state}
          {vision.delegate ? ` · ${vision.delegate}` : ''}
          {vision.state === 'running' ? ` · ${vision.fps.toFixed(0)} fps · ${vision.inferenceMs.toFixed(0)} ms/frame` : ''}
          {vision.error ? ` · ${vision.error}` : ''}
        </p>
      )}
    </div>
  );
}
