import React from 'react';
import { decksApi, type Slide } from '@/lib/decks';

const KIND_LABEL: Record<string, string> = {
  hook: 'Hook', problem: 'Problem', solution: 'Solution', product: 'Product', market: 'Market',
  business_model: 'Business model', traction: 'Traction', competition: 'Competition', team: 'Team',
  financials: 'Financials', ask: 'The ask', closing: 'Closing', title: '',
};

/** A slide at 16:9: the uploaded page image, or a clean template for outline/AI decks. */
export function SlideView({ deckId, slide, version = '', compact = false }: { deckId: string; slide: Slide; version?: string; compact?: boolean }) {
  if (slide.has_image) {
    return (
      <div className="keep-palette aspect-video w-full overflow-hidden rounded-xl bg-white">
        <img src={decksApi.imageUrl(deckId, slide.idx, version)} alt={slide.title || `Slide ${slide.idx + 1}`} className="h-full w-full object-contain" draggable={false} />
      </div>
    );
  }
  return (
    <div
      className={`keep-palette flex aspect-video w-full flex-col justify-center overflow-hidden rounded-xl bg-gradient-to-br from-neutral-50 to-neutral-200 text-neutral-900 ${compact ? 'p-3' : 'p-[6%]'}`}
      role="img"
      aria-label={slide.title || `Slide ${slide.idx + 1}`}
    >
      {slide.kind && KIND_LABEL[slide.kind] && (
        <span className={`font-semibold uppercase tracking-widest text-purple-700 ${compact ? 'text-[6px]' : 'text-[clamp(10px,1.4vw,16px)]'}`}>
          {KIND_LABEL[slide.kind]}
        </span>
      )}
      <h2 className={`font-extrabold leading-tight ${compact ? 'text-[10px]' : 'text-[clamp(18px,3.6vw,48px)]'}`}>{slide.title || 'Untitled slide'}</h2>
      {slide.bullets.length > 0 && (
        <ul className={`mt-[3%] list-disc space-y-1 pl-[1.2em] ${compact ? 'text-[6px]' : 'text-[clamp(12px,2vw,28px)]'}`}>
          {slide.bullets.map((b, i) => (
            <li key={i}>{b}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
