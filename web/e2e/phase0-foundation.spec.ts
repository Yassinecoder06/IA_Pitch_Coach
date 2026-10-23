import { expect, test } from '@playwright/test';
import { configureProviders, GEMINI_KEY, OLLAMA_KEY, PASSPHRASE, unlock } from './helpers';

test.describe.configure({ mode: 'serial' });

test('claim install, coach a typed pitch with a live model, get a structured scorecard', async ({ page }) => {
  test.skip(!GEMINI_KEY && !OLLAMA_KEY, 'needs a live AI key');
  await unlock(page);
  await expect(page.getByRole('navigation', { name: 'Main' })).toBeVisible();
  await expect(page).toHaveURL(/\/coach$/);
  // Before any pitch the insights panel is empty: no placeholder score.
  await expect(page.getByText('Live SOTA Insights')).toBeVisible();
  await expect(page.getByTestId('overall-score')).toHaveCount(0);

  await configureProviders(page);
  await page.getByRole('button', { name: /New Pitch Session/i }).click();
  await page.getByLabel('Pitch text').fill(
    'We built FreightPilot. Small retailers overpay for shipping by 30%. We pool their volume and book carriers ' +
      'automatically. 40 paying stores today, raising $1.5M to reach 500.',
  );
  await page.getByLabel('Send').click();

  await expect(page.getByText(/FreightPilot/).first()).toBeVisible();
  // Streaming coach reply, then the judge's structured score replaces "--".
  await expect(page.getByTestId('overall-score')).toBeVisible({ timeout: 150_000 });
  await expect(page.getByTestId('overall-score')).not.toContainText('--', { timeout: 150_000 });
  const score = Number((await page.getByTestId('overall-score').innerText()).replace(/\D+.*$/s, ''));
  expect(score).toBeGreaterThan(0);
  expect(score).toBeLessThanOrEqual(100);

  // Persisted: reload shows the turn and the stored score.
  await page.reload();
  await expect(page.getByText(/FreightPilot/).first()).toBeVisible();
  await expect(page.getByTestId('overall-score')).not.toContainText('--');
});

test('voice pitch: fake microphone PCM is transcribed by local Whisper', async ({ page }) => {
  await unlock(page);
  await page.getByRole('button', { name: /New Pitch Session/i }).click();
  await page.getByRole('button', { name: 'Start recording' }).click();
  await page.waitForTimeout(16_000); // the fake mic plays the spoken pitch once
  await page.getByRole('button', { name: 'Stop recording' }).click();
  await expect(page.getByText(/retailers|FreightPilot|freight/i).first()).toBeVisible({ timeout: 90_000 });
  await expect(page.getByText(/pools|stores|million/i).first()).toBeVisible();
});

test('profiles: create a PIN-protected profile, data is separated, switching needs the PIN', async ({ page }) => {
  await unlock(page);
  await page.goto('/settings/profiles');
  await page.getByLabel('New profile name').fill('Cofounder');
  await page.getByRole('button', { name: 'Add profile' }).click();
  const row = page.locator('.glass-card', { has: page.locator('input[value="Cofounder"]') });
  await row.getByLabel('4-digit PIN').fill('2468');
  await row.getByRole('button', { name: 'Save PIN' }).click();
  await expect(row.getByRole('button', { name: 'Remove PIN' })).toBeVisible();

  await page.getByLabel(/^Profile .* Switch profile$/).click();
  await page.getByRole('button', { name: 'Switch profile', exact: true }).click();
  await expect(page.getByText('Who is practicing?')).toBeVisible();
  await page.getByRole('button', { name: /Cofounder/ }).click();
  await page.getByLabel('PIN for Cofounder').fill('0000');
  await page.getByRole('button', { name: 'Open' }).click();
  await expect(page.getByRole('alert')).toContainText('Wrong PIN');
  await page.getByLabel('PIN for Cofounder').fill('2468');
  await page.getByRole('button', { name: 'Open' }).click();

  await page.goto('/coach');
  await expect(page.getByText(/FreightPilot/)).toHaveCount(0); // the other profile's sessions are not visible
});

test('lock signs out and the passphrase unlocks again', async ({ page }) => {
  await unlock(page);
  // With several profiles unlocking lands on the picker; pick one, then lock from the app shell.
  if (await page.getByText('Who is practicing?').isVisible()) await page.getByRole('button', { name: 'Me', exact: true }).click();
  await page.getByLabel('Lock Pitch Coach').click();
  await expect(page.getByPlaceholder('Enter passphrase')).toBeVisible();
  await page.getByPlaceholder('Enter passphrase').fill('wrong-passphrase');
  await page.getByRole('button', { name: /Unlock Coach/ }).click();
  await expect(page.getByText('Passphrase is not correct.')).toBeVisible();
  await page.getByPlaceholder('Enter passphrase').fill(PASSPHRASE);
  await page.getByRole('button', { name: /Unlock Coach/ }).click();
  await expect(page.getByRole('navigation', { name: 'Main' }).or(page.getByText('Who is practicing?'))).toBeVisible();
});
