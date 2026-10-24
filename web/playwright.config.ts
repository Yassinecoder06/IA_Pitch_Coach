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
    // E2E_GPU=1 uses the real GPU (ANGLE/EGL); otherwise WebGL falls back to SwiftShader in headless Chrome.
    ...(process.env.E2E_GPU ? ['--ignore-gpu-blocklist', '--enable-gpu', '--use-gl=angle', '--use-angle=gl-egl'] : ['--enable-unsafe-swiftshader']),
  ];
}

/** With E2E_GPU=1 on a hybrid-graphics laptop, PRIME offload puts Chrome's WebGL on the NVIDIA GPU. */
export function chromeEnv(): Record<string, string> {
  const env = Object.fromEntries(Object.entries(process.env).filter((e): e is [string, string] => e[1] !== undefined));
  if (!process.env.E2E_GPU) return env;
  return {
    ...env,
    __NV_PRIME_RENDER_OFFLOAD: '1',
    __GLX_VENDOR_LIBRARY_NAME: 'nvidia',
    __EGL_VENDOR_LIBRARY_FILENAMES: '/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
  };
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
    launchOptions: { args: chromeArgs(), env: chromeEnv() },
  },
});
