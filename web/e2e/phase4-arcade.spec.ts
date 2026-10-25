import path from 'node:path';
import { expect, test, type Page } from '@playwright/test';
import { chromeArgs, chromeEnv, E2E_DIR } from '../playwright.config';
import { unlock } from './helpers';

test.use({ launchOptions: { args: chromeArgs('portrait'), env: chromeEnv() } });
test.describe.configure({ mode: 'serial' });

async function play(page: Page, game: string, seconds: number) {
  await page.goto(`/arcade/${game}`);
  await page.getByRole('radio', { name: 'easy' }).click();
  await page.getByRole('button', { name: 'Play' }).click();
  await expect(page.getByTestId('game-score')).toBeVisible({ timeout: 90_000 });
  await page.waitForTimeout(seconds * 1000);
  await page.screenshot({ path: path.join(E2E_DIR, `${game}.png`) });
  await expect(page.getByTestId('game-over')).toContainText(/Play again/, { timeout: 120_000 });
  return Number((await page.getByTestId('game-over').locator('.text-6xl').innerText()).trim());
}

test('arcade hub shows the daily queue', async ({ page }) => {
  await unlock(page);
  await page.goto('/arcade');
  await expect(page.getByTestId('daily-queue').locator('li')).toHaveCount(3);
});

test('Eye Lock: the portrait looking at the lens charges the ring', async ({ page }) => {
  test.setTimeout(240_000);
  await unlock(page);
  const score = await play(page, 'eye_lock', 10);
  console.log('Eye Lock score:', score);
  expect(score).toBeGreaterThan(100); // at least one full lock on easy
  await expect(page.getByTestId('game-over')).toContainText(/eye contact pct\s*\d+/);
});

test('Pitch Surfer: a natural speaking voice stays on the wave', async ({ page }) => {
  test.setTimeout(240_000);
  await unlock(page);
  const score = await play(page, 'pitch_surfer', 10);
  console.log('Pitch Surfer score:', score);
  expect(score).toBeGreaterThan(0);

  await page.goto('/arcade');
  await expect(page.getByTestId('streak')).toContainText(/[1-9]\d*\s*day streak/);
  await expect(page.getByText(/Best \d+/).first()).toBeVisible();
});
