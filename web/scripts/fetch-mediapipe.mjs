// Copies the MediaPipe WASM runtime and downloads the face/pose models into public/mediapipe,
// so vision runs fully offline at runtime (no CDN). Safe to re-run; existing files are kept.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(HERE, '..', 'public', 'mediapipe');
const WASM_SRC = path.resolve(HERE, '..', 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');

const MODELS = {
  'face_landmarker.task':
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  'pose_landmarker_lite.task':
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
};

fs.mkdirSync(path.join(OUT, 'wasm'), { recursive: true });
fs.mkdirSync(path.join(OUT, 'models'), { recursive: true });

for (const file of fs.readdirSync(WASM_SRC)) {
  fs.copyFileSync(path.join(WASM_SRC, file), path.join(OUT, 'wasm', file));
}

for (const [name, url] of Object.entries(MODELS)) {
  const dest = path.join(OUT, 'models', name);
  if (fs.existsSync(dest) && fs.statSync(dest).size > 0) continue;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed (${res.status}) for ${url}`);
  fs.writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
  console.log(`[mediapipe] downloaded ${name}`);
}
console.log(`[mediapipe] assets ready in ${path.relative(process.cwd(), OUT)}`);
