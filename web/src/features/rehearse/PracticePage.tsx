import React, { useRef, useState } from 'react';
import { Camera, CameraOff, Crosshair, Mic, Play, Square } from 'lucide-react';
import { CameraView } from './components/CameraView';
import { Hud } from './components/Hud';
import { LiveCaptions } from './components/LiveCaptions';
import { useLiveTake } from './useLiveTake';
import { toast } from '@/stores/toast';

/** Free practice: live sensors and captions without a deck; nothing is stored. */
export function PracticePage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [camera, setCamera] = useState(true);
  const [overlay, setOverlay] = useState(true);
  const { phase, error, result, start, stop, calibrate } = useLiveTake();
  const live = phase === 'live';

  return (
    <div className="grid h-full grid-cols-1 gap-4 overflow-y-auto p-4 lg:grid-cols-[minmax(0,1fr)_380px]">
      <section className="flex min-h-0 flex-col gap-3">
        <header className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h1 className="text-lg font-bold">Live practice</h1>
            <p className="text-xs text-neutral-400">
              Camera analysis runs in your browser; only numbers are kept. Audio goes to your local Whisper.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setCamera(c => !c)}
              disabled={live}
              aria-pressed={camera}
              className="flex items-center gap-1.5 rounded-xl border border-white/10 px-3 py-2 text-xs hover:bg-white/5 disabled:opacity-50"
            >
              {camera ? <Camera className="h-4 w-4" /> : <CameraOff className="h-4 w-4" />} Camera {camera ? 'on' : 'off'}
            </button>
            {live && camera && (
              <button
                onClick={async () => {
                  toast('Look straight into the camera lens for 3 seconds...');
                  const base = await calibrate();
                  toast(base ? 'Eye contact calibrated.' : 'No face found; calibration skipped.', base ? 'info' : 'error');
                }}
                className="flex items-center gap-1.5 rounded-xl border border-white/10 px-3 py-2 text-xs hover:bg-white/5"
              >
                <Crosshair className="h-4 w-4" /> Calibrate eye contact
              </button>
            )}
            {!live ? (
              <button
                onClick={() => void start({ camera, mic: true, video: videoRef.current })}
                disabled={phase === 'starting' || phase === 'finishing'}
                className="flex items-center gap-1.5 rounded-xl bg-purple-600 px-4 py-2 text-sm font-semibold hover:bg-purple-500 disabled:opacity-50"
              >
                <Play className="h-4 w-4" /> {phase === 'starting' ? 'Starting...' : 'Start practice'}
              </button>
            ) : (
              <button
                onClick={() => void stop()}
                className="flex items-center gap-1.5 rounded-xl bg-rose-600 px-4 py-2 text-sm font-semibold hover:bg-rose-500"
              >
                <Square className="h-4 w-4" /> Stop
              </button>
            )}
          </div>
        </header>

        {error && <p className="rounded-xl bg-rose-500/10 px-3 py-2 text-sm text-rose-300" role="alert">{error}</p>}

        <div className="grid min-h-0 flex-1 gap-3 md:grid-cols-2">
          <div className="space-y-2">
            <CameraView ref={videoRef} overlay={overlay} active={live && camera} />
            <label className="flex items-center gap-2 text-xs text-neutral-400">
              <input type="checkbox" checked={overlay} onChange={e => setOverlay(e.target.checked)} /> Show tracking overlay
            </label>
          </div>
          <div className="min-h-[200px]">
            <LiveCaptions />
          </div>
        </div>

        {phase === 'finishing' && <p className="text-sm text-neutral-400">Finishing the transcript...</p>}
        {result && (
          <div className="rounded-2xl border border-white/10 bg-neutral-900/60 p-4 text-sm" data-testid="practice-summary">
            <h2 className="font-semibold">Run summary</h2>
            <p className="mt-1 text-neutral-400">
              {result.duration.toFixed(0)} s · {result.words.length} words
              {result.sensors.eye_contact_pct !== null && ` · eye contact ${result.sensors.eye_contact_pct}%`}
              {result.sensors.posture_avg !== null && ` · posture ${result.sensors.posture_avg}`}
              {result.sensors.pitch_range_avg_st !== null && ` · pitch range ${result.sensors.pitch_range_avg_st} st`}
            </p>
            <p className="mt-2 text-xs text-neutral-500">
              Want a full report, slide timing and history? Use a deck in the Rehearsal Studio.
            </p>
          </div>
        )}
      </section>

      <aside className="space-y-3">
        <div className="flex items-center gap-2 text-xs text-neutral-400">
          <Mic className={`h-4 w-4 ${live ? 'text-rose-400' : ''}`} /> {live ? 'Recording' : 'Not recording'}
        </div>
        <Hud />
      </aside>
    </div>
  );
}
