import React, { useState, useEffect } from 'react';
import { Timer, Play, Pause, RotateCcw, X } from 'lucide-react';

interface DrillTimerProps {
  onClose?: () => void;
}

const PRESETS = [
  { label: '60s Elevator', seconds: 60 },
  { label: '3m Seed Pitch', seconds: 180 },
  { label: '5m Demo Day', seconds: 300 },
];

export function DrillTimer({ onClose }: DrillTimerProps) {
  const [totalSeconds, setTotalSeconds] = useState(60);
  const [remaining, setRemaining] = useState(60);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    let interval: any = null;
    if (isRunning && remaining > 0) {
      interval = setInterval(() => {
        setRemaining(prev => prev - 1);
      }, 1000);
    } else if (remaining === 0 && isRunning) {
      setIsRunning(false);
      // Play brief notification sound or visual alert
    }
    return () => clearInterval(interval);
  }, [isRunning, remaining]);

  const selectPreset = (secs: number) => {
    setIsRunning(false);
    setTotalSeconds(secs);
    setRemaining(secs);
  };

  const reset = () => {
    setIsRunning(false);
    setRemaining(totalSeconds);
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const progress = ((totalSeconds - remaining) / totalSeconds) * 100;

  return (
    <div className="glass-card flex items-center justify-between rounded-2xl p-3 border border-purple-500/20">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-purple-500/10 text-purple-400">
          <Timer className="h-4 w-4" />
        </div>
        <div>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400 block">
            Pitch Drill Timer
          </span>
          <span className={`text-lg font-mono font-bold ${remaining <= 10 && remaining > 0 ? 'text-amber-400 animate-pulse' : remaining === 0 ? 'text-rose-400' : 'text-white'}`}>
            {formatTime(remaining)}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-1.5">
        <div className="hidden sm:flex items-center gap-1 bg-neutral-900/60 p-1 rounded-xl border border-white/5">
          {PRESETS.map(p => (
            <button
              key={p.seconds}
              onClick={() => selectPreset(p.seconds)}
              className={`rounded-lg px-2 py-1 text-[11px] font-medium transition-all ${
                totalSeconds === p.seconds
                  ? 'bg-purple-600 text-white shadow-sm'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>

        <button
          onClick={() => setIsRunning(!isRunning)}
          className={`flex h-8 w-8 items-center justify-center rounded-xl text-white shadow-md transition-all ${
            isRunning
              ? 'bg-amber-600 hover:bg-amber-500'
              : 'bg-purple-600 hover:bg-purple-500'
          }`}
          title={isRunning ? 'Pause' : 'Start'}
        >
          {isRunning ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5 ml-0.5" />}
        </button>

        <button
          onClick={reset}
          className="flex h-8 w-8 items-center justify-center rounded-xl border border-white/10 bg-white/5 text-neutral-400 hover:bg-white/10 hover:text-white"
          title="Reset"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>

        {onClose && (
          <button
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-xl text-neutral-500 hover:text-neutral-300"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    </div>
  );
}
