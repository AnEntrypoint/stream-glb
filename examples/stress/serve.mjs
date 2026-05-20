#!/usr/bin/env node
// Tiny static file server with HTTP Range support. Serves from the repo
// root so `/runtime/...`, `/examples/...` and `/output_*/...` (baked
// assets) are all reachable from one origin. The /assets-list.json
// endpoint reflects every `output_*` directory under ASSETS_DIR (default
// = repo root).

import { createServer } from 'node:http';
import { stat, open, readdir } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const REPO_ROOT = resolve(__dirname, '..', '..');
const ASSETS_DIR = process.env.ASSETS_DIR
  ? resolve(process.env.ASSETS_DIR)
  : REPO_ROOT;
const PORT = Number(process.env.PORT) || 5180;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'application/javascript; charset=utf-8',
  '.mjs':  'application/javascript; charset=utf-8',
  '.json': 'application/json',
  '.glb':  'model/gltf-binary',
  '.webp': 'image/webp',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.css':  'text/css',
};

async function tryResolve(urlPath) {
  // assets dir first (output_*), then repo root for everything else.
  for (const base of [ASSETS_DIR, REPO_ROOT]) {
    const full = normalize(join(base, urlPath));
    if (!full.startsWith(base)) continue;
    const s = await stat(full).catch(() => null);
    if (s && s.isFile()) return { full, s };
  }
  return null;
}

createServer(async (req, res) => {
  try {
    let urlPath = decodeURIComponent(req.url.split('?')[0]);
    if (urlPath === '/' || urlPath === '') {
      // Redirect to the stress demo so relative imports in stress.html
      // (./stress.js, ./test-streaming/...) resolve against /examples/stress/.
      res.writeHead(302, { Location: '/examples/stress/' });
      res.end();
      return;
    }
    if (urlPath === '/examples/stress' || urlPath === '/examples/stress/') urlPath = '/examples/stress/stress.html';
    // Auto-resolve dir/ → dir/index.html for any directory.
    if (urlPath.endsWith('/')) urlPath += 'index.html';
    if (urlPath === '/assets-list.json') {
      const entries = await readdir(ASSETS_DIR, { withFileTypes: true });
      const dirs = entries
        .filter((e) => e.isDirectory() && e.name.startsWith('output_'))
        .map((e) => e.name)
        .sort();
      res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
      res.end(JSON.stringify(dirs));
      return;
    }

    const hit = await tryResolve(urlPath);
    if (!hit) { res.writeHead(404).end('not found'); return; }
    const { full, s } = hit;

    const mime = MIME[extname(full).toLowerCase()] || 'application/octet-stream';
    res.setHeader('Content-Type', mime);
    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('Cache-Control', 'no-store');

    const range = req.headers.range;
    if (range) {
      const m = /^bytes=(\d*)-(\d*)$/.exec(range);
      if (m) {
        const start = m[1] === '' ? Math.max(0, s.size - Number(m[2])) : Number(m[1]);
        const end = m[2] === '' ? s.size - 1 : Math.min(Number(m[2]), s.size - 1);
        if (start <= end && start < s.size) {
          res.writeHead(206, {
            'Content-Range': `bytes ${start}-${end}/${s.size}`,
            'Content-Length': end - start + 1,
          });
          const fh = await open(full, 'r');
          fh.createReadStream({ start, end }).pipe(res);
          return;
        }
      }
    }

    res.setHeader('Content-Length', s.size);
    const fh = await open(full, 'r');
    fh.createReadStream().pipe(res);
  } catch (e) {
    console.error(e);
    res.writeHead(500).end(String(e));
  }
}).listen(PORT, '127.0.0.1', () => {
  console.log(`[serve] http://127.0.0.1:${PORT}/`);
  console.log(`[serve] repo=${REPO_ROOT}`);
  console.log(`[serve] assets=${ASSETS_DIR}`);
});
