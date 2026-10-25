import fs from 'node:fs';
import path from 'node:path';
import { expect, test, type Page } from '@playwright/test';
import { chromeArgs, chromeEnv, E2E_DIR } from '../playwright.config';
import { configureProviders, fullPageShot, GEMINI_KEY, OLLAMA_KEY, unlock } from './helpers';

test.use({ launchOptions: { args: chromeArgs('portrait'), env: chromeEnv() } });
test.describe.configure({ mode: 'serial' });

async function freeTake(page: Page, name: string, seconds: number) {
  await unlock(page);
  if (GEMINI_KEY || OLLAMA_KEY) await configureProviders(page);
  await page.goto('/rehearse');
  await page.getByLabel('Deck').selectOption('');
  await page.getByLabel('Target length').fill('0.5');
  await page.getByLabel('Take name').fill(name);
  await page.getByRole('button', { name: 'Start rehearsal' }).click();
  await expect(page.getByTestId('slide-counter')).toHaveText('Free talk', { timeout: 60_000 });
  await page.waitForTimeout(seconds * 1000);
  await page.getByRole('button', { name: 'Stop' }).click();
  await expect(page).toHaveURL(/\/report\/[\w-]+$/, { timeout: 300_000 });
  await expect(page.getByTestId('score-overall')).not.toHaveText('--');
}

test('two takes for the history', async ({ page }) => {
  test.setTimeout(600_000);
  await freeTake(page, 'History take 1', 8);
  await freeTake(page, 'History take 2', 14);
});

test('history trend, compare two takes, goals', async ({ page }) => {
  await unlock(page);
  await page.goto('/history');
  const rows = page.getByTestId('attempts').locator('tbody tr');
  await expect(rows.filter({ hasText: 'History take' })).toHaveCount(2);
  await expect(page.getByTestId('trend').locator('svg circle').first()).toBeVisible();

  await page.getByLabel('Select History take 1 to compare').check();
  await page.getByLabel('Select History take 2 to compare').check();
  await page.getByRole('button', { name: /Compare selected/ }).click();
  await expect(page).toHaveURL(/\/history\/compare\?a=.+&b=.+/);
  await expect(page.getByTestId('compare-scores')).toContainText('Overall');
  await expect(page.getByTestId('compare-facts')).toContainText('Duration');
  await expect(page.getByTestId('compare-cues')).toContainText(/resolved/i);
  await fullPageShot(page, path.join(E2E_DIR, 'compare.png'));

  await page.goto('/history');
  await page.getByLabel('Goal metric').selectOption('eye_contact_pct');
  await page.getByLabel('Goal condition').selectOption('gte');
  await page.getByLabel('Target').fill('50');
  await page.getByRole('button', { name: 'Add goal' }).click();
  await expect(page.getByTestId('goals')).toContainText(/Eye contact \(%\) at least 50/);
  await expect(page.getByTestId('goals')).toContainText(/Latest \d+/);
});

test('printable report exports to PDF', async ({ page }) => {
  await unlock(page);
  await page.goto('/history');
  await page.getByTestId('attempts').getByRole('link', { name: 'History take 2' }).click();
  await page.getByRole('link', { name: 'PDF report' }).click();
  await expect(page).toHaveURL(/\/print$/);
  await expect(page.getByTestId('print')).toBeVisible();
  await expect(page.locator('.paper')).toHaveCSS('background-color', 'rgb(255, 255, 255)');
  await page.screenshot({ path: path.join(E2E_DIR, 'print.png'), fullPage: true });
  await page.emulateMedia({ media: 'print' });
  const pdf = path.join(E2E_DIR, 'report.pdf');
  await page.pdf({ path: pdf, format: 'A4', printBackground: true });
  const bytes = fs.readFileSync(pdf);
  expect(bytes.subarray(0, 4).toString()).toBe('%PDF');
  expect(bytes.length).toBeGreaterThan(20_000);
  console.log('PDF bytes:', bytes.length);
});

test('privacy: turning recordings off is saved', async ({ page }) => {
  await unlock(page);
  await page.goto('/settings/privacy');
  const keep = page.getByLabel(/Keep audio recordings/);
  await expect(keep).toBeChecked();
  await page.getByLabel('Recording retention').selectOption('30');
  await expect(page.getByText('Privacy settings saved.')).toBeVisible();
  await keep.uncheck();
  await page.reload();
  await expect(page.getByLabel(/Keep audio recordings/)).not.toBeChecked();
  await page.getByLabel(/Keep audio recordings/).check(); // restore for other specs
});
