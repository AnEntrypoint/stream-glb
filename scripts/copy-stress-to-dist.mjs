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

// Patch stress.html to point at ./stress.js and the runtime via the
// site-absolute path that GitHub Pages serves (./../runtime/...).
const stressHtmlPath = path.join(dist, 'stress/index.html');
const { readFile } = await import('node:fs/promises');
let html = await readFile(stressHtmlPath, 'utf8');
// stress.js currently does `import { ModelPool } from '/runtime/model-pool.js'`.
// On GH Pages under /stream-glb/stress/, that resolves to the wrong place;
// rewrite to a relative path in the copy.
let stressJs = await readFile(path.join(dist, 'stress/stress.js'), 'utf8');
stressJs = stressJs.replace(
  "from '/runtime/model-pool.js'",
  "from '../runtime/model-pool.js'"
);
await writeFile(path.join(dist, 'stress/stress.js'), stressJs);
// Disable static-asset-list fetch on the deployed page since GH Pages
// doesn't have one. Users can paste a custom URL.
stressJs = stressJs.replace(
  "fetch('/assets-list.json')",
  "fetch('./assets-list.json')"
);
await writeFile(path.join(dist, 'stress/stress.js'), stressJs);

console.log('[copy-stress] dist/stress/ ready');
