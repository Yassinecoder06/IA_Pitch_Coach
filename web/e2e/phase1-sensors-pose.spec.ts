import { expect, test } from '@playwright/test';
import { chromeArgs } from '../playwright.config';
import { unlock } from './helpers';

// Fake camera: MediaPipe's pose test image (side-on face, arms stretched out).
test.use({ launchOptions: { args: chromeArgs('pose') } });

test('live practice: a speaker looking away does not get eye-contact credit', async ({ page }) => {
  await unlock(page);
  await page.goto('/practice');
  await page.getByRole('button', { name: 'Start practice' }).click();
  await expect(page.getByTestId('vision-status')).toContainText('running', { timeout: 90_000 });
  await page.waitForTimeout(5000);
  await expect(page.getByTestId('hud-eye')).not.toContainText('Good');
  await expect(page.getByTestId('hud-posture')).not.toContainText('Arms crossed');
  console.log('eye:', await page.getByTestId('hud-eye').innerText(), '| posture:', await page.getByTestId('hud-posture').innerText());
  await page.getByRole('button', { name: 'Stop' }).click();
  await expect(page.getByTestId('practice-summary')).toBeVisible({ timeout: 60_000 });
});
