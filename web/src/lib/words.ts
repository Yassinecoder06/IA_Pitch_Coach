/** Whisper splits tokens like "$1" ".5" or "it" "'s"; these attach to the previous word without a space. */
const ATTACH = /^(?:[.,!?;:%)\]]|'[a-z]|\.\d|n't)/i;

export const attachesToPrevious = (token: string) => ATTACH.test(token.trim());

export function joinWords(words: { w: string }[]): string {
  let out = '';
  for (const { w } of words) {
    const tok = w.trim();
    if (!tok) continue;
    out += !out || attachesToPrevious(tok) ? tok : ` ${tok}`;
  }
  return out;
}
