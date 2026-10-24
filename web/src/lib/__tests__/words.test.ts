import { describe, expect, it } from 'vitest';
import { joinWords } from '../words';

describe('joinWords', () => {
  it('matches the backend join (timing.join_words)', () => {
    const w = ['raising', '$1', '.5', 'million,', 'it', "'s", '30', '%', 'growth.'].map(t => ({ w: t }));
    expect(joinWords(w)).toBe("raising $1.5 million, it's 30% growth.");
  });
});
