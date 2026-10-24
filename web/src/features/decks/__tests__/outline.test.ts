import { describe, expect, it } from 'vitest';
import { parseOutline } from '../importDeck';

describe('parseOutline', () => {
  it('splits paragraphs into slides with bullets, notes and budgets', () => {
    const slides = parseOutline(`# Shipping is broken [25s]
- Retailers overpay 30%
* Rates are opaque
> Open with the pain.
> Pause after the number.

FreightPilot`);
    expect(slides).toHaveLength(2);
    expect(slides[0]).toMatchObject({
      title: 'Shipping is broken',
      time_budget_sec: 25,
      bullets: ['Retailers overpay 30%', 'Rates are opaque'],
      notes: 'Open with the pain. Pause after the number.',
    });
    expect(slides[1]).toMatchObject({ title: 'FreightPilot', bullets: [], time_budget_sec: null });
  });

  it('ignores extra blank lines', () => {
    expect(parseOutline('\n\nA\n\n\n\nB\n')).toHaveLength(2);
  });
});
