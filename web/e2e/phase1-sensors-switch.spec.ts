import { expect, test } from '@playwright/test';
import { chromeArgs, chromeEnv } from '../playwright.config';
import { unlock } from './helpers';

// Fake camera: the portrait for 10 s, then an empty frame for 6 s, looping. The HUD must follow
// each change, not just settle on a single still image.
// One loop of the clip is 16 s; each state must show up within a loop plus margin.
const CYCLE = 35_000;

test.use({ launchOptions: { args: chromeArgs('switch'), env: chromeEnv() } });

test('live practice: the HUD reacts when the speaker leaves the frame and comes back', async ({ page }) => {
  await unlock(page);
  await page.goto('/practice');
  await page.getByRole('button', { name: 'Start practice' }).click();
  await expect(page.getByTestId('vision-status')).toContainText('running', { timeout: 90_000 });

  const eye = page.getByTestId('hud-eye');
  const posture = page.getByTestId('hud-posture');
  for (let cycle = 0; cycle < 2; cycle++) {
    await expect(eye).toContainText('Good', { timeout: CYCLE });
    await expect(posture).toContainText('Arms crossed', { timeout: CYCLE });
    await expect(eye).toContainText('No face', { timeout: CYCLE });
    await expect(posture).not.toContainText('Arms crossed', { timeout: CYCLE });
    console.log(`cycle ${cycle + 1}: good -> no face -> ...`);
  }
  await expect(eye).toContainText('Good', { timeout: CYCLE });

  await page.getByRole('button', { name: 'Stop' }).click();
  const summary = page.getByTestId('practice-summary');
  await expect(summary).toBeVisible({ timeout: 60_000 });
  console.log('summary:', await summary.innerText());
});
