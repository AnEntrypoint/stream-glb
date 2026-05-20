#!/usr/bin/env node
// After flatspace builds the static site into ./dist, copy the live stress
// demo + runtime into ./dist/stress/ so GitHub Pages serves a runnable demo
// at /stress/. Bakes are NOT shipped — the deployed demo points at the
// user's own asset host (or fails the fetch and the HUD shows 'no assets').
//
// For an actually-running deployed demo, host bakes elsewhere and edit
// /stress/stress.js to fetch your asset-list URL.

import { cp, mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const dist = path.join(repoRoot, 'dist');
if (!existsSync(dist)) {
  console.error('[copy-stress] dist/ missing — run flatspace build first');
  process.exit(1);
}

await mkdir(path.join(dist, 'stress'), { recursive: true });
await cp(path.join(repoRoot, 'examples/stress/stress.html'), path.join(dist, 'stress/index.html'));
await cp(path.join(repoRoot, 'examples/stress/stress.js'), path.join(dist, 'stress/stress.js'));
await mkdir(path.join(dist, 'runtime'), { recursive: true });
await cp(path.join(repoRoot, 'runtime/model-pool.js'), path.join(dist, 'runtime/model-pool.js'));
await cp(path.join(repoRoot, 'runtime/lod-worker.js'), path.join(dist, 'runtime/lod-worker.js'));

// Vendor: three.js + GLTFLoader + meshopt + three-vrm + anentrypoint-design.
// These live at /vendor/ locally (served by examples/stress/serve.mjs which
// serves the repo root) and at /stream-glb/vendor/ on GH Pages. The lod-worker
// resolves them via `new URL('../vendor/...', import.meta.url)` which works
// in both locations because it's relative to the worker URL. stress.html
// uses absolute `/vendor/...` in source; we rewrite to `../vendor/...` for
// dist so the deployed importmap works under /stream-glb/.
await cp(path.join(repoRoot, 'vendor'), path.join(dist, 'vendor'), { recursive: true });

// stress.js source-of-truth uses `from '/runtime/model-pool.js'` — on
// GH Pages under /stream-glb/stress/ that absolute path is wrong; rewrite
// to a relative path in the deployed copy. The runtime layout under
// dist/runtime/ mirrors the repo runtime/ so '../runtime/...' resolves.
// We do NOT rewrite '/assets-list.json' — stress.js handles 404 by
// falling back to the remote manifest at anentrypoint.github.io/assets/.
const { readFile } = await import('node:fs/promises');
let stressJs = await readFile(path.join(dist, 'stress/stress.js'), 'utf8');
stressJs = stressJs.replace(
  "from '/runtime/model-pool.js'",
  "from '../runtime/model-pool.js'"
);
await writeFile(path.join(dist, 'stress/stress.js'), stressJs);

// Rewrite the importmap in dist/stress/index.html: source uses absolute
// `/vendor/...` (valid under local serve.mjs which serves the repo root),
// dist needs `../vendor/...` so GH Pages resolves it relative to
// /stream-glb/stress/index.html → /stream-glb/vendor/...
let stressHtml = await readFile(path.join(dist, 'stress/index.html'), 'utf8');
stressHtml = stressHtml.replace(/"\/vendor\//g, '"../vendor/');
await writeFile(path.join(dist, 'stress/index.html'), stressHtml);

console.log('[copy-stress] dist/stress/ ready (with /vendor → ../vendor rewrite)');
