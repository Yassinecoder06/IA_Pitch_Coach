import path from 'node:path';
import { expect, test } from '@playwright/test';
import { E2E_DIR } from '../playwright.config';
import { configureProviders, GEMINI_KEY, OLLAMA_KEY, unlock } from './helpers';

test.describe.configure({ mode: 'serial' });

test('outline deck: create, edit notes, save, persists', async ({ page }) => {
  await unlock(page);
  await page.goto('/decks');
  await page.getByRole('button', { name: 'Write outline' }).click();
  await page.getByLabel('Deck title').fill('E2E Outline Deck');
  await page.getByRole('button', { name: 'Create deck' }).click();
  await expect(page).toHaveURL(/\/decks\/[\w-]+$/);
  await expect(page.getByLabel('Slides').getByText('3. Raising $1.5M seed')).toBeVisible();

  await page.getByLabel('Slides').getByText('2. FreightPilot pools the volume').click();
  await page.getByLabel(/Speaker notes/).fill('Say the saving per parcel.');
  await page.getByRole('button', { name: 'Save' }).click();
  await expect(page.getByText('Deck saved.')).toBeVisible();
  await page.reload();
  await page.getByLabel('Slides').getByText('2. FreightPilot pools the volume').click();
  await expect(page.getByLabel(/Speaker notes/)).toHaveValue('Say the saving per parcel.');
});

for (const file of ['sample.pdf', 'sample.pptx']) {
  test(`import ${file}: pages rendered in the browser and stored as slide images`, async ({ page }) => {
    await unlock(page);
    await page.goto('/decks');
    await page.getByRole('button', { name: 'Upload PDF / PPTX' }).click();
    await page.getByTestId('deck-file').setInputFiles(path.join(E2E_DIR, file));
    await expect(page).toHaveURL(/\/decks\/[\w-]+$/, { timeout: 120_000 });
    const thumbs = page.getByLabel('Slides').locator('img');
    await expect(thumbs).toHaveCount(3);
    await expect.poll(() => thumbs.first().evaluate(img => (img as HTMLImageElement).naturalWidth)).toBeGreaterThan(1000);
    await expect(page.getByLabel('Slides')).toContainText('Shipping eats retail margins');
    if (file.endsWith('pptx')) {
      // Speaker notes come from the PPTX itself.
      await expect(page.getByLabel(/Speaker notes/)).toHaveValue("Open with a store owner's story.");
    }
  });
}

test('AI deck generation and slide critique with a live model', async ({ page }) => {
  test.skip(!GEMINI_KEY && !OLLAMA_KEY, 'needs a live AI key');
  test.setTimeout(400_000);
  await unlock(page);
  await configureProviders(page);
  await page.goto('/decks');
  await page.getByRole('button', { name: 'Generate with AI' }).click();
  await page.getByLabel('Company name').fill('FreightPilot');
  await page.getByLabel('One-liner').fill('Pools shipping volume so small retailers get big-brand carrier rates');
  await page.getByLabel('Length').selectOption('2');
  await page.getByLabel('Details').fill('40 paying stores, 30% average saving, raising $1.5M seed, team ex-Flexport.');
  await page.getByRole('button', { name: 'Generate deck' }).click();
  await expect(page).toHaveURL(/\/decks\/[\w-]+$/, { timeout: 240_000 });
  const count = await page.getByLabel('Slides').locator('li').count();
  expect(count).toBeGreaterThanOrEqual(3);
  console.log('AI deck slides:', await page.getByLabel('Slides').innerText());

  await page.getByRole('button', { name: 'Critique my slides' }).click();
  await expect(page.getByTestId('critique')).toBeVisible({ timeout: 240_000 });
  console.log('critique:', (await page.getByTestId('critique').innerText()).slice(0, 300));
});
