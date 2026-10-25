/**
 * Context-aware filler detection, mirroring backend/analysis/fillers.py.
 * Rules come from shared/fillers.json; both sides are tested on shared/filler_cases.json.
 */
import RULES from '@shared/fillers.json';

const ALWAYS = new Set<string>(RULES.always);
const DISCOURSE = new Set<string>(RULES.discourse);
const PHRASES: string[][] = RULES.phrases;
const YOU_KNOW_NOT_BEFORE = new Set<string>(RULES.you_know_not_before);
const LIKE_FILLER_AFTER = new Set<string>(RULES.like_filler_after);
const LIKE_NOT_AFTER = new Set<string>(RULES.like_not_after);
const SENTENCE_MARKERS = new Set<string>(RULES.sentence_markers);

const TOKEN_RE = /[A-Za-z0-9]+(?:['’][A-Za-z]+)?/g;

interface Token {
  text: string;
  start: number;
  end: number;
  commaBefore: boolean;
  punctAfter: string;
  sentenceStart: boolean;
}

export interface FillerMatch {
  filler: string;
  start: number; // char offsets
  end: number;
}

function tokenize(text: string): Token[] {
  const tokens: Token[] = [];
  let prevEnd = 0;
  for (const m of text.matchAll(TOKEN_RE)) {
    const start = m.index ?? 0;
    const end = start + m[0].length;
    const between = text.slice(prevEnd, start);
    const after = text.charAt(end);
    tokens.push({
      text: m[0].toLowerCase().replace('’', "'"),
      start,
      end,
      commaBefore: between.includes(',') || between.includes(';'),
      punctAfter: ',.?!;'.includes(after) && after !== '' ? after : '',
      sentenceStart: tokens.length === 0 || /[.!?]/.test(between),
    });
    prevEnd = end;
  }
  return tokens;
}

const isPlainFiller = (t?: Token) => !!t && (ALWAYS.has(t.text) || DISCOURSE.has(t.text));

function matchPhrase(tokens: Token[], i: number): string[] | null {
  for (const phrase of PHRASES) {
    const n = phrase.length;
    if (i + n > tokens.length) continue;
    if (phrase.every((w, k) => tokens[i + k].text === w) && tokens.slice(i, i + n - 1).every(t => !t.punctAfter)) {
      return phrase;
    }
  }
  return null;
}

function likeIsFiller(tok: Token, prev?: Token): boolean {
  if (prev && LIKE_NOT_AFTER.has(prev.text)) return false;
  if (tok.commaBefore || tok.punctAfter === ',') return true;
  return !!prev && LIKE_FILLER_AFTER.has(prev.text);
}

function rightIsFiller(tok: Token, next?: Token): boolean {
  const tagQuestion = tok.punctAfter === '?' && (!next || next.sentenceStart);
  return tagQuestion || (tok.commaBefore && tok.punctAfter === ',');
}

function markerIsFiller(tok: Token, prev?: Token, next?: Token): boolean {
  const atClauseStart = tok.sentenceStart || isPlainFiller(prev);
  return atClauseStart && (tok.punctAfter === ',' || isPlainFiller(next));
}

export function findFillers(text: string): FillerMatch[] {
  const tokens = tokenize(text);
  const matches: FillerMatch[] = [];
  let i = 0;
  while (i < tokens.length) {
    const tok = tokens[i];
    const prev = tokens[i - 1];
    const next = tokens[i + 1];

    const phrase = matchPhrase(tokens, i);
    if (phrase) {
      const n = phrase.length;
      const following = tokens[i + n];
      const last = tokens[i + n - 1];
      const isQuestion = phrase.join(' ') === 'you know' && following && YOU_KNOW_NOT_BEFORE.has(following.text) && !last.punctAfter;
      if (!isQuestion) {
        matches.push({ filler: phrase.join(' '), start: tok.start, end: last.end });
        i += n;
        continue;
      }
    }

    const w = tok.text;
    if (ALWAYS.has(w) || DISCOURSE.has(w)) matches.push({ filler: w, start: tok.start, end: tok.end });
    else if (w === 'like' && likeIsFiller(tok, prev)) matches.push({ filler: w, start: tok.start, end: tok.end });
    else if (w === 'right' && rightIsFiller(tok, next)) matches.push({ filler: w, start: tok.start, end: tok.end });
    else if (SENTENCE_MARKERS.has(w) && w !== 'right' && markerIsFiller(tok, prev, next))
      matches.push({ filler: w, start: tok.start, end: tok.end });
    i += 1;
  }
  return matches;
}

export function countFillers(text: string): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const m of findFillers(text)) counts[m.filler] = (counts[m.filler] ?? 0) + 1;
  return counts;
}

/** Splits text into plain and filler segments for highlighting. */
export function segmentFillers(text: string): { text: string; filler: string | null }[] {
  const out: { text: string; filler: string | null }[] = [];
  let pos = 0;
  for (const m of findFillers(text)) {
    if (m.start > pos) out.push({ text: text.slice(pos, m.start), filler: null });
    out.push({ text: text.slice(m.start, m.end), filler: m.filler });
    pos = m.end;
  }
  if (pos < text.length) out.push({ text: text.slice(pos), filler: null });
  return out;
}
