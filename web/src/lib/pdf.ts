import * as pdfjs from 'pdfjs-dist';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

export interface RenderedPage {
  image: Blob; // JPEG
  text: string;
}

export const MAX_PAGES = 80;

/** Renders every page of a PDF to a JPEG (1600 px wide) and extracts its text, in the browser. */
export async function renderPdf(
  source: ArrayBuffer | string,
  onProgress?: (done: number, total: number) => void,
): Promise<RenderedPage[]> {
  const doc = await pdfjs.getDocument(typeof source === 'string' ? { url: source, withCredentials: true } : { data: source }).promise;
  const total = Math.min(doc.numPages, MAX_PAGES);
  const pages: RenderedPage[] = [];
  for (let n = 1; n <= total; n++) {
    const page = await doc.getPage(n);
    const base = page.getViewport({ scale: 1 });
    const viewport = page.getViewport({ scale: 1600 / base.width });
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(viewport.width);
    canvas.height = Math.round(viewport.height);
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    await page.render({ canvasContext: ctx, viewport }).promise;
    const image = await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob(b => (b ? resolve(b) : reject(new Error('Could not render page'))), 'image/jpeg', 0.85),
    );
    const content = await page.getTextContent();
    const text = content.items
      .map(item => ('str' in item ? item.str : ''))
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();
    pages.push({ image, text });
    page.cleanup();
    onProgress?.(n, total);
  }
  await doc.destroy();
  return pages;
}

/** Best-effort slide title: the first short line of text on the page. */
export function guessTitle(text: string, fallback: string): string {
  const first = text.split(/(?<=[.!?])\s|\s{2,}/)[0]?.trim() ?? '';
  return first && first.length <= 80 ? first : fallback;
}
