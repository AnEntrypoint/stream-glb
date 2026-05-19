#!/usr/bin/env node
// Bulk-bake every .glb/.vrm under one or more source dirs through
// bake-progressive.mjs. Each asset gets its own output_<basename> directory
// so the runtime can load any of them by directory name. Per-asset
// success/fail is reported. Recurses into subdirectories.
//
// Usage:
//   node tools/bake-all.mjs                          # bakes ./models
//   node tools/bake-all.mjs <dir> [<dir>...]         # bakes given dirs
//   PARALLEL=8 node tools/bake-all.mjs <dir>         # parallel worker count
//
// Output dirs always land in examples/local-progressive/output_<basename>.
// Collisions on basename are resolved by suffixing _2, _3, ...

import { readdir, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');

const SOURCE_DIRS = process.argv.length > 2
  ? process.argv.slice(2)
  : [path.join(repoRoot, 'models')];
const OUTPUT_BASE = path.join(repoRoot, 'examples/local-progressive');
const PARALLEL = Math.max(1, parseInt(process.env.PARALLEL || '4', 10));
const SKIP_IF_EXISTS = process.env.SKIP_EXISTING !== '0';

async function walk(dir, out = []) {
  let entries;
  try { entries = await readdir(dir, { withFileTypes: true }); }
  catch { return out; }
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) await walk(full, out);
    else if (e.isFile() && /\.(glb|vrm)$/i.test(e.name)) out.push(full);
  }
  return out;
}

function pickOutDir(base) {
  let name = `output_${base}`;
  let suffix = 1;
  while (existsSync(path.join(OUTPUT_BASE, name)) && !SKIP_IF_EXISTS) {
    suffix++;
    name = `output_${base}_${suffix}`;
  }
  return path.join(OUTPUT_BASE, name);
}

async function main() {
  const all = [];
  for (const root of SOURCE_DIRS) {
    console.log(`[bake-all] scanning ${root}`);
    const found = await walk(root);
    console.log(`[bake-all]   found ${found.length} assets`);
    for (const p of found) all.push(p);
  }
  console.log(`[bake-all] total assets: ${all.length}, parallel=${PARALLEL}`);

  const results = { ok: [], failed: [], skipped: [] };
  let cursor = 0;
  const t0 = Date.now();

  async function worker(id) {
    while (true) {
      const myIdx = cursor++;
      if (myIdx >= all.length) return;
      const inPath = all[myIdx];
      const base = path.basename(inPath, path.extname(inPath));
      const outDir = pickOutDir(base);
      const outGlb = path.join(outDir, 'model.progressive.glb');
      if (SKIP_IF_EXISTS && existsSync(outGlb)) {
        results.skipped.push({ name: base });
        if (myIdx % 25 === 0) console.log(`[bake-all] [w${id}] ${myIdx+1}/${all.length} skip ${base}`);
        continue;
      }
      const startMs = Date.now();
      try {
        await runBake(inPath, outDir);
        const dt = Date.now() - startMs;
        let rootSize = 0;
        try { rootSize = (await stat(outGlb)).size; } catch {}
        results.ok.push({ name: base, outDir, ms: dt, rootMB: (rootSize/1024/1024).toFixed(2) });
        console.log(`[bake-all] [w${id}] ${myIdx+1}/${all.length} ${base} (${(rootSize/1024/1024).toFixed(2)} MB, ${dt}ms)`);
      } catch (e) {
        results.failed.push({ name: base, error: e.message });
        console.error(`[bake-all] [w${id}] ${myIdx+1}/${all.length} ${base} FAILED: ${e.message.slice(0, 120)}`);
      }
    }
  }
  await Promise.all(Array.from({ length: PARALLEL }, (_, i) => worker(i)));
  const totalDt = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`\n[bake-all] DONE in ${totalDt}s: ${results.ok.length} ok, ${results.skipped.length} skipped, ${results.failed.length} failed`);
  if (results.failed.length) {
    for (const f of results.failed.slice(0, 50)) console.log(`  x ${f.name}: ${f.error.slice(0, 200)}`);
    if (results.failed.length > 50) console.log(`  ... +${results.failed.length - 50} more`);
  }
}

function runBake(inPath, outDir) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      process.execPath,
      [path.join(__dirname, 'bake-progressive.mjs'), inPath, outDir],
      { stdio: ['ignore', 'pipe', 'pipe'] }
    );
    let stderr = '';
    child.stdout.on('data', () => {});
    child.stderr.on('data', (d) => { stderr += d.toString(); });
    child.on('error', reject);
    const killTimer = setTimeout(() => {
      try { child.kill('SIGKILL'); } catch {}
      reject(new Error('timeout 180s'));
    }, 180_000);
    child.on('exit', (code) => {
      clearTimeout(killTimer);
      if (code === 0) resolve();
      else reject(new Error(`exit ${code}${stderr ? ': ' + stderr.split('\n').slice(-3).join(' ').trim() : ''}`));
    });
  });
}

main().catch((e) => { console.error(e); process.exit(1); });
