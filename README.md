# stream-glb

Local-first progressive GLB / VRM streaming for [three.js](https://threejs.org). Bake any glb/vrm into a per-LOD progressive bundle, stream over HTTP range requests, render 1000+ unique assets at 35+ fps with GPU-side frustum culling and a 4-worker decode pool. Skinning and morph targets preserved.

Witnessed: **1100 unique-asset entities @ 35.8 fps · 154k tris rendered · 51 draw calls · `pool.update` 0.3 ms/frame.**

## Install

```sh
npm i @anentrypoint/stream-glb three
```

## Bake assets

```sh
PARALLEL=8 node tools/bake-all.mjs ./my-models
```

Walks the directory tree, runs `tools/bake-progressive.mjs` per glb/vrm, emits `output_<name>/model.progressive.glb` plus sibling per-LOD `.glb`/`.webp` files.

## Use the runtime

```js
import * as THREE from 'three';
import { ModelPool } from '@anentrypoint/stream-glb';

const pool = new ModelPool({
  scene, renderer, camera,
  targetFps: 60,
  byteBudget: 256 * 1024 * 1024,
  workerCount: 4,
});

const entity = pool.spawn('/output_avatar/model.progressive.glb', {
  position: [0, 0, 0],
  static: true,
});
scene.add(entity.root);

// in your render loop:
pool.update();
renderer.render(scene, camera);
```

## What's in the box

- `runtime/model-pool.js` — `ModelPool` + `Entity` + `InstancedSlot` with GPU frustum cull, per-frame adaptive FPS/VRAM budget controller, per-asset InstancedMesh sharing, scene-graph pruning for FAR-tier instanced entities.
- `runtime/lod-worker.js` — module Worker that fetches + parses sibling LOD GLBs off the main thread via GLTFLoader + MeshoptDecoder, posts back transferable typed arrays.
- `tools/bake-progressive.mjs` — bakes one glb/vrm into a progressive bundle (root + sibling LODs). Supports Draco-compressed inputs (`KHR_draco_mesh_compression`), meshopt encoding, KHR_mesh_quantization, vertex-color and unskinned LOD tail stages.
- `tools/bake-all.mjs` — recursive parallel bulk baker. `PARALLEL=N` controls worker count.
- `examples/stress/` — 1000-entity stress test page (`stress.html`) with live HUD showing fps, draws, triangles, per-phase ms, tier counts.

## License

MIT
