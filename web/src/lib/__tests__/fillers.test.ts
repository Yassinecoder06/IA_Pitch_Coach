import { describe, expect, it } from 'vitest';
import CASES from '@shared/filler_cases.json';
import { countFillers, segmentFillers } from '../fillers';

describe('filler detection (shared cases with the Python backend)', () => {
  it.each(CASES.map(c => [c.text, c.expected] as const))('%s', (text, expected) => {
    expect(countFillers(text)).toEqual(expected);
  });

  it('segments text for highlighting without losing characters', () => {
    const text = 'So, um, we grew, like, a lot.';
    const segs = segmentFillers(text);
    expect(segs.map(s => s.text).join('')).toBe(text);
    expect(segs.filter(s => s.filler).map(s => s.filler)).toEqual(['so', 'um', 'like']);
  });
});
