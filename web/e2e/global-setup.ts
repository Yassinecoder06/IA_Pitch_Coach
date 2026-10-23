import { spawn, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
import { E2E_DIR, E2E_PORT } from '../playwright.config';

const ROOT = path.resolve(HERE, '..', '..');
const PYTHON = path.join(ROOT, '.venv', 'bin', 'python');

export const PITCH_TEXT =
  'Hi, I am the founder of FreightPilot. Small retailers overpay for shipping by about thirty percent. ' +
  'Our app pools their volume and books carriers automatically. We have forty paying stores, ' +
  'and we are raising one point five million dollars to reach five hundred stores.';

/** Boots the real backend on a scratch database and records the printed setup code. */
export default async function globalSetup() {
  fs.rmSync(E2E_DIR, { recursive: true, force: true });
  fs.mkdirSync(E2E_DIR, { recursive: true });

  // Spoken pitch for Chrome's fake microphone (48 kHz mono WAV, rendered with the local Piper voice).
  execFileSync(PYTHON, ['-I', path.join(HERE, 'make_wav.py'), path.join(E2E_DIR, 'pitch.wav'), PITCH_TEXT], {
    cwd: ROOT,
    stdio: 'inherit',
  });

  const server = spawn(
    path.join(ROOT, '.venv', 'bin', 'uvicorn'),
    ['backend.main:app', '--host', '127.0.0.1', '--port', String(E2E_PORT)],
    {
      cwd: ROOT,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        DATABASE_URL: `sqlite:///${path.join(E2E_DIR, 'e2e.db')}`,
        DATA_DIR: E2E_DIR,
        SECRET_KEY: 'e2e-secret-key',
        PRELOAD_STT_MODEL: 'true',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );
  const log = fs.createWriteStream(path.join(E2E_DIR, 'server.log'));
  server.stdout.pipe(log);
  server.stderr.pipe(log);
  fs.writeFileSync(path.join(E2E_DIR, 'server.pid'), String(server.pid));

  const code = await new Promise<string>((resolve, reject) => {
    let out = '';
    const timer = setTimeout(() => reject(new Error(`Backend did not start:\n${out}`)), 120_000);
    const onData = (chunk: Buffer) => {
      out += chunk.toString();
      const m = out.match(/>>>\s+([A-Z0-9]{4}-[A-Z0-9]{4})\s+<<</);
      if (m && out.includes('[Ready]')) {
        clearTimeout(timer);
        resolve(m[1]);
      }
    };
    server.stdout.on('data', onData);
    server.on('exit', c => reject(new Error(`Backend exited with ${c}:\n${out}`)));
  });
  fs.writeFileSync(path.join(E2E_DIR, 'setup-code.txt'), code);
}
