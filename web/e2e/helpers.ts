import fs from 'node:fs';
import path from 'node:path';
import { expect, type Page } from '@playwright/test';
import { E2E_DIR } from '../playwright.config';

export const PASSPHRASE = 'E2E-Passphrase-123';
export const GEMINI_KEY = process.env.GEMINI_API_KEY;
export const OLLAMA_KEY = process.env.OLLAMA_API_KEY;
export const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-3.8-flash';
export const OLLAMA_MODEL = process.env.OLLAMA_CLOUD_MODEL || 'beam';

export const setupCode = () => fs.readFileSync(path.join(E2E_DIR, 'setup-code.txt'), 'utf8').trim();

/** Claims the install on first run, or unlocks it afterwards; ends on the coach page. */
export async function unlock(page: Page, options: { pickDefault?: boolean } = {}) {
  await page.goto('/');
  const codeInput = page.getByPlaceholder('XXXX-XXXX');
  const passInput = page.getByPlaceholder(/At least 8 characters|Enter passphrase/);
  const nav = page.getByRole('navigation', { name: 'Main' });
  await expect(passInput.or(nav).or(page.getByText('Who is practicing?'))).toBeVisible();
  if (!(await passInput.isVisible())) {
    // already unlocked in this browser context
  } else if (await codeInput.isVisible()) {
    await codeInput.fill(setupCode());
    await passInput.fill(PASSPHRASE);
    await page.getByPlaceholder('Re-enter passphrase').fill(PASSPHRASE);
    await page.getByRole('button', { name: /Claim & Unlock/ }).click();
  } else {
    await passInput.fill(PASSPHRASE);
    await page.getByRole('button', { name: /Unlock Coach/ }).click();
  }
  // Wait for the login round-trip: either the app shell or the profile picker.
  await expect(page.getByRole('navigation', { name: 'Main' }).or(page.getByText('Who is practicing?'))).toBeVisible();
  if (options.pickDefault !== false && (await page.getByText('Who is practicing?').isVisible())) {
    await page.getByRole('button', { name: 'Me', exact: true }).click();
    await expect(page.getByRole('navigation', { name: 'Main' })).toBeVisible();
  }
}

/** Registers the live providers through the API: Ollama Cloud (beam) primary, Gemini fallback. Idempotent. */
export async function configureProviders(page: Page) {
  const refs: string[] = [];
  const existing = await (await page.request.get('/api/ai/providers')).json();
  const ensure = async (kind: string, apiKey: string) => {
    const found = existing.providers.find((p: any) => p.kind === kind);
    if (found) return found.id as string;
    const state = await (await page.request.post('/api/ai/providers', { data: { kind, apiKey } })).json();
    return state.providers.find((p: any) => p.kind === kind).id as string;
  };
  if (OLLAMA_KEY) refs.push(`${await ensure('ollama_cloud', OLLAMA_KEY)}:${OLLAMA_MODEL}`);
  if (GEMINI_KEY) refs.push(`${await ensure('gemini', GEMINI_KEY)}:${GEMINI_MODEL}`);
  const res = await page.request.put('/api/ai/preferences', { data: { chat_models: refs.slice(0, 1), fallback_models: refs.slice(1) } });
  expect(res.ok()).toBeTruthy();
  return refs;
}

/** Full-page screenshot of an app page whose content scrolls inside fixed-height containers. */
export async function fullPageShot(page: Page, file: string) {
  await page.addStyleTag({
    content: 'html, body, #root, #root * { overflow: visible !important; } .h-screen, .h-full { height: auto !important; }',
  });
  await page.waitForTimeout(300);
  await page.screenshot({ path: file, fullPage: true });
}
