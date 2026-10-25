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
  // A server left over from an interrupted run would answer instead of the fresh one.
  const stalePid = path.join(E2E_DIR, 'server.pid');
  if (fs.existsSync(stalePid)) {
    try {
      process.kill(Number(fs.readFileSync(stalePid, 'utf8')), 'SIGTERM');
      await new Promise(r => setTimeout(r, 1500));
    } catch {
      /* not running */
    }
  }
  fs.rmSync(E2E_DIR, { recursive: true, force: true });
  fs.mkdirSync(E2E_DIR, { recursive: true });

  // Spoken pitch for Chrome's fake microphone (48 kHz mono WAV, rendered with the local Piper voice).
  execFileSync(PYTHON, ['-I', path.join(HERE, 'make_wav.py'), path.join(E2E_DIR, 'pitch.wav'), PITCH_TEXT], {
    cwd: ROOT,
    stdio: 'inherit',
  });

  execFileSync(PYTHON, ['-I', path.join(HERE, 'make_deck.py'), E2E_DIR], { cwd: ROOT, stdio: 'inherit' });

  // Fake-camera clips from MediaPipe's own test images (downloaded, never committed).
  for (const name of ['portrait', 'pose']) {
    const jpg = path.join(E2E_DIR, `${name}.jpg`);
    const res = await fetch(`https://storage.googleapis.com/mediapipe-assets/${name}.jpg`);
    if (!res.ok) throw new Error(`Could not download ${name}.jpg (${res.status})`);
    fs.writeFileSync(jpg, Buffer.from(await res.arrayBuffer()));
    execFileSync('ffmpeg', [
      '-loglevel', 'error', '-y', '-loop', '1', '-i', jpg, '-t', '6',
      '-vf', 'scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,fps=15',
      '-pix_fmt', 'yuv420p', path.join(E2E_DIR, `${name}.y4m`),
    ]);
  }

  // Speaker steps out of frame and comes back: 10 s portrait, then 6 s empty room (Chrome loops it).
  execFileSync('ffmpeg', [
    '-loglevel', 'error', '-y', '-loop', '1', '-t', '10', '-i', path.join(E2E_DIR, 'portrait.jpg'),
    '-f', 'lavfi', '-i', 'color=c=0x303030:s=640x480:r=15:d=6',
    '-filter_complex',
    '[0:v]scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,fps=15,format=yuv420p,setsar=1[a];' +
      '[1:v]format=yuv420p,setsar=1[b];[a][b]concat=n=2:v=1[v]',
    '-map', '[v]', '-pix_fmt', 'yuv420p', path.join(E2E_DIR, 'switch.y4m'),
  ]);

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
      // "Uvicorn running on" is printed only after the port is bound; "[Ready]" comes earlier.
      if (m && out.includes('Uvicorn running on')) {
        clearTimeout(timer);
        resolve(m[1]);
      }
    };
    server.stdout.on('data', onData);
    server.stderr.on('data', onData); // uvicorn logs to stderr
    server.on('exit', c => reject(new Error(`Backend exited with ${c}:\n${out}`)));
  });
  fs.writeFileSync(path.join(E2E_DIR, 'setup-code.txt'), code);
}
