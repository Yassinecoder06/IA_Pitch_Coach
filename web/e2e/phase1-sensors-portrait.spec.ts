import { expect, test } from '@playwright/test';
import { chromeArgs, chromeEnv } from '../playwright.config';
import { unlock } from './helpers';

// Fake camera: MediaPipe's portrait test image (facing the lens, arms crossed).
test.use({ launchOptions: { args: chromeArgs('portrait'), env: chromeEnv() } });

test('live practice: vision worker tracks the face and posture, voice and captions update', async ({ page }) => {
  await unlock(page);
  await page.goto('/practice');
  await page.getByRole('button', { name: 'Start practice' }).click();

  const status = page.getByTestId('vision-status');
  await expect(status).toContainText('running', { timeout: 90_000 });
  console.log('vision status:', await status.innerText());

  // The portrait looks straight at the camera with crossed arms.
  await expect(page.getByTestId('hud-eye')).toContainText(/Good/, { timeout: 30_000 });
  await expect(page.getByTestId('hud-posture')).toContainText('Arms crossed', { timeout: 30_000 });

  // Voice: the fake mic plays the spoken pitch; pitch range and captions appear.
  await expect(page.getByTestId('hud-pitch')).toContainText(/\d+(\.\d)? st/, { timeout: 30_000 });
  await expect(page.getByTestId('captions')).toContainText(/retailers|freight|shipping/i, { timeout: 60_000 });
  await expect(page.getByTestId('hud-pace')).toContainText(/\d+ wpm/, { timeout: 60_000 });

  const statusText = await status.innerText();
  const ms = Number(statusText.match(/(\d+) ms\/frame/)?.[1] ?? '999');
  console.log('inference ms/frame:', ms, '|', statusText);

  await page.waitForTimeout(4000);
  await page.getByRole('button', { name: 'Stop' }).click();
  const summary = page.getByTestId('practice-summary');
  await expect(summary).toContainText(/eye contact \d+(\.\d)?%/, { timeout: 60_000 });
  await expect(summary).toContainText(/posture \d+/);
  console.log('summary:', await summary.innerText());
});
