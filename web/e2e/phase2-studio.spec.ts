import { expect, test } from '@playwright/test';
import path from 'node:path';
import { chromeArgs, chromeEnv, E2E_DIR } from '../playwright.config';
import { configureProviders, fullPageShot, GEMINI_KEY, OLLAMA_KEY, unlock } from './helpers';

test.use({ launchOptions: { args: chromeArgs('portrait'), env: chromeEnv() } });

test('rehearse a deck slide by slide, get the full report, then practise Q&A', async ({ page }) => {
  test.setTimeout(480_000);
  await unlock(page);
  if (GEMINI_KEY || OLLAMA_KEY) await configureProviders(page);

  // Deck from the outline sample
  await page.goto('/decks');
  await page.getByRole('button', { name: 'Write outline' }).click();
  await page.getByLabel('Deck title').fill('Studio Deck');
  await page.getByRole('button', { name: 'Create deck' }).click();
  await page.getByRole('button', { name: 'Rehearse this deck' }).click();

  await expect(page.getByLabel('Deck')).toHaveValue(/[\w-]{36}/);
  await page.getByLabel('Target length').fill('0.5');
  await page.getByLabel('Take name').fill('E2E take');
  await page.getByRole('button', { name: 'Start rehearsal' }).click();

  await expect(page.getByTestId('slide-counter')).toHaveText('Slide 1 / 3', { timeout: 60_000 });
  await expect(page.getByTestId('vision-status')).toContainText('running', { timeout: 90_000 });
  await page.waitForTimeout(6000);
  await page.keyboard.press('ArrowRight');
  await expect(page.getByTestId('slide-counter')).toHaveText('Slide 2 / 3');
  await page.waitForTimeout(6000);
  await page.keyboard.press('Space');
  await expect(page.getByTestId('slide-counter')).toHaveText('Slide 3 / 3');
  await page.waitForTimeout(6000);
  await page.screenshot({ path: path.join(E2E_DIR, 'presenter.png') });
  await page.getByRole('button', { name: 'Stop' }).click();

  // Analysis then report
  await expect(page).toHaveURL(/\/report\/[\w-]+$/, { timeout: 300_000 });
  await expect(page.getByTestId('score-overall')).not.toHaveText('--');
  await expect(page.getByTestId('transcript')).toContainText(/retailers|shipping|freight/i);
  await expect(page.getByTestId('slide-table').locator('tbody tr')).toHaveCount(3);
  await expect(page.getByTestId('score-eye_contact')).not.toHaveText('n/a'); // camera was on
  await expect(page.getByTestId('score-pace')).not.toHaveText('n/a');
  if (GEMINI_KEY || OLLAMA_KEY) {
    await expect(page.getByTestId('verdict')).toBeVisible();
    console.log('verdict:', (await page.getByTestId('verdict').innerText()).slice(0, 500));
  }
  await expect(page.locator('audio')).toHaveAttribute('src', /\/api\/rehearsals\/.+\/audio/);
  console.log('overall score:', await page.getByTestId('score-overall').innerText());
  await page.setViewportSize({ width: 1440, height: 900 });
  await fullPageShot(page, path.join(E2E_DIR, 'report.png'));

  if (GEMINI_KEY || OLLAMA_KEY) {
    await page.getByRole('button', { name: /Practice investor Q&A/ }).click();
    await expect(page).toHaveURL(/\/coach\?session=/);
    await expect(page.getByText(/^Q&A: E2E take/).first()).toBeVisible();
  }
});
