// Stress demo for ModelPool tier system. Spawns unique-asset entities across
// a wide grid, runs orbiting camera, shows live perf + tier counts.

import * as THREE from 'three';
import { ModelPool } from '/runtime/model-pool.js';

// Asset list fetched from server /assets-list.json at boot. Server walks
// the examples/local-progressive directory and emits every output_* dir.
let ASSET_DIRS = [];
const ASSET_DIRS_READY = fetch('/assets-list.json')
  .then((r) => r.json())
  .then((list) => { ASSET_DIRS = list; console.log(`[stress] ${list.length} assets discovered`); return list; })
  .catch((e) => { console.error('[stress] asset-list fetch failed', e); ASSET_DIRS = []; });

const canvas = document.getElementById('c');
const hud = document.getElementById('hud');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(1);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x181820);
scene.add(new THREE.HemisphereLight(0xffffff, 0x222233, 1.0));
const dir = new THREE.DirectionalLight(0xffffff, 1.2);
dir.position.set(20, 30, 20);
scene.add(dir);
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);
camera.position.set(30, 18, 30);
camera.lookAt(0, 1, 0);

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

const pool = new ModelPool({
  scene, renderer, camera,
  targetFps: 60,
  byteBudget: 256 * 1024 * 1024,
  maxConcurrentFetches: 6,
});
window.__pool = pool;

const proxies = new Set();

async function spawnUnique(n) {
  await ASSET_DIRS_READY;
  if (!ASSET_DIRS.length) {
    console.error('[stress] no assets available to spawn');
    return;
  }
  // Distribute entities across a square grid; each entity picks an asset
  // from the full ASSET_DIRS list (modulo), so up to len(ASSET_DIRS) of
  // them are unique.
  const side = Math.ceil(Math.sqrt(n));
  const spacing = 1.5;
  let count = 0;
  for (let row = 0; row < side && count < n; row++) {
    for (let col = 0; col < side && count < n; col++) {
      const x = (col - side / 2) * spacing;
      const z = (row - side / 2) * spacing;
      const asset = ASSET_DIRS[count % ASSET_DIRS.length];
      const proxy = pool.spawn(`${asset}/model.progressive.glb`, {
        position: [x, 0, z],
        rotation: [0, (count * 0.137) % (Math.PI * 2), 0],
        static: true,
      });
      scene.add(proxy.root);
      proxies.add(proxy);
      count++;
    }
  }
  console.log(`[stress] spawned ${count} entities`);
}

document.querySelectorAll('#panel button[data-n]').forEach((btn) => {
  btn.addEventListener('click', () => spawnUnique(+btn.dataset.n));
});
document.getElementById('clear').addEventListener('click', () => {
  for (const p of proxies) p.dispose();
  proxies.clear();
});
document.getElementById('target-fps').addEventListener('change', (e) => {
  pool.targetFps = +e.target.value;
});
document.getElementById('byte-budget').addEventListener('change', (e) => {
  pool.byteBudget = +e.target.value * 1024 * 1024;
});

let orbitT = 0;
let zoomPhase = 0;
function tick() {
  if (document.getElementById('orbit-cam').checked) {
    orbitT += 0.003;
    // Fly-through path: when zoom-cycle is on we pulse from far (r=60) to
    // very close (r=3, INSIDE the crowd) so all three tiers get exercised.
    // The HERO tier needs r<10 to see entities at >200 screen-px.
    let r = 30;
    if (document.getElementById('zoom-cycle').checked) {
      zoomPhase += 0.008;
      // 3..60 — sweep through the crowd, getting up close mid-cycle.
      r = 30 + Math.cos(zoomPhase) * 27;
    }
    camera.position.x = Math.cos(orbitT) * r;
    camera.position.z = Math.sin(orbitT) * r;
    camera.position.y = 6 + Math.sin(orbitT * 0.7) * 4;
    camera.lookAt(0, 1, 0);
  }
  pool.update();
  renderer.render(scene, camera);
  const s = pool.getStats();
  hud.innerHTML = `
    <b>FPS</b> ${s.fps.toFixed(1)} (target ${pool.targetFps})<br>
    <b>entities</b> ${s.entities} <span class="tier">HERO ${s.hero||0} MID ${s.mid||0} FAR ${s.far||0}</span><br>
    <b>draws</b> ${s.drawCalls} <b>ceiling</b> ${s.ceilingLod ?? 'auto'} <b>midPx</b> ${pool.midPx.toFixed(0)} <b>heroCap</b> ${pool.heroCap}<br>
    <b>bytes</b> ${(s.bytes/1024/1024).toFixed(1)}/${(pool.byteBudget/1024/1024).toFixed(0)} MB
    <b>assets</b> ${s.assets} <b>inFlight</b> ${s.inFlight}<br>
    <b>tri</b> ${(renderer.info.render.triangles/1000).toFixed(1)}k<br>
    <b>pool.update</b> ${(s.msTotal||0).toFixed(2)}ms (frustum ${(s.msFrustum||0).toFixed(2)} entities ${(s.msEntities||0).toFixed(2)})
  `;
  requestAnimationFrame(tick);
}
tick();
