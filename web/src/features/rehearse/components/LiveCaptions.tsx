import React, { useEffect, useRef } from 'react';
import { useSensors } from '@/stores/sensors';
import { segmentFillers } from '@/lib/fillers';
import { joinWords } from '@/lib/words';

export function LiveCaptions({ placeholder = 'Start speaking: your words appear here a moment after each pause.' }: { placeholder?: string }) {
  const words = useSensors(s => s.words);
  const endRef = useRef<HTMLDivElement | null>(null);
  const text = joinWords(words);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'nearest' });
  }, [text]);

  return (
    <div className="h-full overflow-y-auto rounded-2xl border border-white/10 bg-neutral-950/60 p-4 text-sm leading-relaxed" data-testid="captions">
      {text ? (
        segmentFillers(text).map((seg, i) =>
          seg.filler ? (
            <mark key={i} className="rounded bg-amber-500/20 px-0.5 text-amber-300" title={`Filler: ${seg.filler}`}>
              {seg.text}
            </mark>
          ) : (
            <span key={i}>{seg.text}</span>
          ),
        )
      ) : (
        <span className="text-neutral-500">{placeholder}</span>
      )}
      <div ref={endRef} />
    </div>
  );
}
