import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
import { defineConfig } from '@playwright/test';

export const E2E_PORT = Number(process.env.E2E_PORT || 8765);
export const E2E_DIR = path.resolve(HERE, 'test-results', 'e2e-runtime');

/** Chrome flags for fake media; `video` names a clip prepared by global-setup (portrait | pose). */
export function chromeArgs(video?: 'portrait' | 'pose'): string[] {
  return [
    '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream',
    `--use-file-for-fake-audio-capture=${path.join(E2E_DIR, 'pitch.wav')}%noloop`,
    ...(video ? [`--use-file-for-fake-video-capture=${path.join(E2E_DIR, `${video}.y4m`)}`] : []),
    '--enable-unsafe-swiftshader', // WebGL in headless Chrome for the GPU delegate
  ];
}

export default defineConfig({
  testDir: './e2e',
  timeout: 180_000,
  expect: { timeout: 60_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  globalSetup: './e2e/global-setup.ts',
  globalTeardown: './e2e/global-teardown.ts',
  use: {
    baseURL: `http://127.0.0.1:${E2E_PORT}`,
    channel: 'chrome',
    headless: true,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    permissions: ['microphone', 'camera'],
    launchOptions: { args: chromeArgs() },
  },
});
