import fs from 'node:fs';
import path from 'node:path';
import { E2E_DIR } from '../playwright.config';

export default async function globalTeardown() {
  const pidFile = path.join(E2E_DIR, 'server.pid');
  if (fs.existsSync(pidFile)) {
    try {
      process.kill(Number(fs.readFileSync(pidFile, 'utf8')), 'SIGTERM');
    } catch {
      /* already stopped */
    }
  }
}
