import fs from 'node:fs';
import path from 'node:path';
import { expect, type Page } from '@playwright/test';
import { E2E_DIR } from '../playwright.config';

export const PASSPHRASE = 'E2E-Passphrase-123';
export const GEMINI_KEY = process.env.GEMINI_API_KEY;
export const OLLAMA_KEY = process.env.OLLAMA_API_KEY;
export const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-3.8-flash';
export const OLLAMA_MODEL = process.env.OLLAMA_CLOUD_MODEL || 'gpt-oss:20b';

export const setupCode = () => fs.readFileSync(path.join(E2E_DIR, 'setup-code.txt'), 'utf8').trim();

/** Claims the install on first run, or unlocks it afterwards; ends on the coach page. */
export async function unlock(page: Page) {
  await page.goto('/');
  const codeInput = page.getByPlaceholder('XXXX-XXXX');
  const passInput = page.getByPlaceholder(/At least 8 characters|Enter passphrase/);
  await expect(passInput).toBeVisible();
  if (await codeInput.isVisible()) {
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
}

/** Registers the live providers through the API: Gemini primary, Ollama Cloud fallback. */
export async function configureProviders(page: Page) {
  const refs: string[] = [];
  if (GEMINI_KEY) {
    const state = await (await page.request.post('/api/ai/providers', { data: { kind: 'gemini', apiKey: GEMINI_KEY } })).json();
    refs.push(`${state.providers.find((p: any) => p.kind === 'gemini').id}:${GEMINI_MODEL}`);
  }
  if (OLLAMA_KEY) {
    const state = await (await page.request.post('/api/ai/providers', { data: { kind: 'ollama_cloud', apiKey: OLLAMA_KEY } })).json();
    refs.push(`${state.providers.find((p: any) => p.kind === 'ollama_cloud').id}:${OLLAMA_MODEL}`);
  }
  const res = await page.request.put('/api/ai/preferences', { data: { chat_models: refs.slice(0, 1), fallback_models: refs.slice(1) } });
  expect(res.ok()).toBeTruthy();
  return refs;
}
