import { decksApi, type Deck, type SlideInput } from '@/lib/decks';
import { guessTitle, renderPdf } from '@/lib/pdf';

export type ImportProgress = (message: string) => void;

const baseName = (name: string) => name.replace(/\.(pdf|pptx)$/i, '').replace(/[_-]+/g, ' ').trim() || 'Imported deck';

/** PDF: rendered in the browser. PPTX: converted to PDF on the server first (keeps titles and speaker notes). */
export async function importDeckFile(file: File, onProgress: ImportProgress): Promise<Deck> {
  const isPptx = /\.pptx$/i.test(file.name);
  let pptxSlides: { title: string; text: string; notes: string }[] = [];
  let source: ArrayBuffer | string;
  if (isPptx) {
    onProgress('Converting PowerPoint...');
    const converted = await decksApi.convertPptx(file);
    pptxSlides = converted.slides;
    source = converted.pdf_url;
  } else if (/\.pdf$/i.test(file.name) || file.type === 'application/pdf') {
    source = await file.arrayBuffer();
  } else {
    throw new Error('Choose a PDF or PPTX file.');
  }

  const pages = await renderPdf(source, (done, total) => onProgress(`Rendering page ${done} of ${total}...`));
  if (!pages.length) throw new Error('That file has no pages.');

  const slides: Partial<SlideInput>[] = pages.map((p, i) => {
    const meta = pptxSlides[i];
    return {
      title: meta?.title || guessTitle(p.text, `Slide ${i + 1}`),
      text: meta?.text || p.text,
      notes: meta?.notes || '',
      bullets: [],
    };
  });
  const deck = await decksApi.create(baseName(file.name), isPptx ? 'pptx' : 'pdf', slides);
  for (let i = 0; i < pages.length; i++) {
    onProgress(`Uploading slide ${i + 1} of ${pages.length}...`);
    await decksApi.uploadImage(deck.id, i, pages[i].image);
  }
  return decksApi.get(deck.id);
}

/**
 * Outline format: blank line between slides; first line is the title, "- " lines are bullets,
 * "> " lines are speaker notes, "[30s]" in the title sets the slide's time budget.
 */
export function parseOutline(text: string): Partial<SlideInput>[] {
  return text
    .split(/\n\s*\n/)
    .map(block => block.split('\n').map(l => l.trim()).filter(Boolean))
    .filter(lines => lines.length > 0)
    .map(lines => {
      let title = lines[0].replace(/^#+\s*/, '');
      let budget: number | null = null;
      const m = title.match(/\[(\d+)\s*s\]/i);
      if (m) {
        budget = Number(m[1]);
        title = title.replace(m[0], '').trim();
      }
      const bullets = lines.slice(1).filter(l => /^[-*•]\s+/.test(l)).map(l => l.replace(/^[-*•]\s+/, ''));
      const notes = lines.slice(1).filter(l => l.startsWith('>')).map(l => l.replace(/^>\s?/, '')).join(' ');
      return { title, bullets, notes, text: '', kind: null, time_budget_sec: budget };
    });
}
