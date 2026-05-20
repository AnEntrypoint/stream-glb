// ModelPool — managed LOD streaming for skinned + morph-target meshes.
//
// Responsibilities:
//  - Load a baked progressive asset bundle (root GLB + sibling LOD files +
//    LOCAL_progressive extras blob) at most ONCE per source URL.
//  - Share BufferGeometry and Texture instances across every Entity that
//    spawns from the same asset, so 1000 LANMOWERs reuse one geometry per LOD
//    rather than allocating 1000 copies.
//  - Spawn lightweight Entity handles (SkinnedMesh / Mesh wrappers) wired to
//    the right shared resources for their current LOD.
//  - Run one per-frame update that walks every live Entity, picks an LOD by
//    screen-space density + global ceiling, evicts/fetches as needed.
//  - Emit events ('ready', 'lod-changed', 'evicted', 'budget-pressure',
//    'fps') so application code can react without polling.
//
// Phase A scope: load-once, instance-share, event-driven Entity API.
// Phases B & C bolt onto this without changing the public surface.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { VRMLoaderPlugin } from '@pixiv/three-vrm';
// (impostor/billboard system removed: FAR-tier entities use the existing
// per-asset unskinned-LOD InstancedMesh which preserves 3D shape and works
// for any geometry — characters, terrain chunks, props — not just things
// that read OK as flat sprites.)

// --- scratch objects (per-frame; never alloc in hot path) -----------------
const _tmpV3 = new THREE.Vector3();
const _tmpV3b = new THREE.Vector3();
const _tmpSphere = new THREE.Sphere();
const _zeroMatrix = new THREE.Matrix4().set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

// Monotonic clock for LRU touch timestamps. performance.now() in browser,
// process.hrtime-ish fallback elsewhere — only needs to be monotonic-ish.
const _now = (typeof performance !== 'undefined' && performance.now)
  ? () => performance.now()
  : () => Date.now();

// --- shared GLTFLoaders ---------------------------------------------------
// Two flavors: one with the VRM plugin (root loads), one without (sibling
// LOD loads — the siblings carry no VRM extension blob, and the plugin's
// MToon prep has side effects on attribute layout we don't want).
function _makeLoader(includeVrm) {
  const l = new GLTFLoader();
  l.setMeshoptDecoder(MeshoptDecoder);
  if (includeVrm) l.register((parser) => new VRMLoaderPlugin(parser));
  return l;
}

// --- GLB composition for streaming sub-LODs -------------------------------
// Given a minimal glTF JSON spec (single buffer of byteLength = bin.length)
// and a Uint8Array BIN payload, build a valid GLB byte buffer. Padding rules
// per the GLB spec: JSON chunk padded with 0x20 (spaces) to 4-byte align;
// BIN chunk padded with 0x00 to 4-byte align.
function _composeSubGLB(jsonSpec, binBytes) {
  const enc = new TextEncoder();
  let jsonBytes = enc.encode(JSON.stringify(jsonSpec));
  const jsonPad = (4 - (jsonBytes.byteLength % 4)) % 4;
  if (jsonPad) {
    const padded = new Uint8Array(jsonBytes.byteLength + jsonPad);
    padded.set(jsonBytes, 0);
    for (let i = 0; i < jsonPad; i++) padded[jsonBytes.byteLength + i] = 0x20;
    jsonBytes = padded;
  }
  const binPad = (4 - (binBytes.byteLength % 4)) % 4;
  const binLen = binBytes.byteLength + binPad;
  const total = 12 /* header */ + 8 + jsonBytes.byteLength + 8 + binLen;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  // GLB header: magic 'glTF', version 2, length.
  dv.setUint32(0, 0x46546c67, true); // 'glTF'
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  // JSON chunk
  let off = 12;
  dv.setUint32(off, jsonBytes.byteLength, true); off += 4;
  dv.setUint32(off, 0x4e4f534a, true); off += 4; // 'JSON'
  out.set(jsonBytes, off); off += jsonBytes.byteLength;
  // BIN chunk
  dv.setUint32(off, binLen, true); off += 4;
  dv.setUint32(off, 0x004e4942, true); off += 4; // 'BIN\0'
  out.set(binBytes, off);
  // trailing pad bytes default to 0 already (Uint8Array init).
  return out;
}

// --- tiny EventEmitter ----------------------------------------------------
class Emitter {
  constructor() { this._listeners = new Map(); }
  on(ev, fn) {
    let s = this._listeners.get(ev);
    if (!s) { s = new Set(); this._listeners.set(ev, s); }
    s.add(fn);
    return () => s.delete(fn);
  }
  emit(ev, payload) {
    const s = this._listeners.get(ev);
    if (!s) return;
    for (const fn of s) {
      try { fn(payload); } catch (e) { console.error(`[ModelPool] listener for ${ev} threw`, e); }
    }
  }
}

// --- InstancedPool: one shared InstancedMesh per (asset, lod) -------------
// For the unskinned LOD tier we don't need per-entity skeletons or per-entity
// SkinnedMesh shells; the mesh is in bind pose and only its TRANSFORM differs
// across entities. Wrapping them all in one InstancedMesh collapses N draw
// calls into 1, which is the only realistic path to 1000+ entities on
// commodity hardware.
class InstancedSlot {
  constructor(pool, asset, meshDescIdx, lodIdx, geo, material) {
    this.pool = pool;
    this.asset = asset;
    this.meshDescIdx = meshDescIdx;
    this.lodIdx = lodIdx;
    this.geometry = geo;
    this.material = material;
    this.capacity = 32; // grow as needed
    // Per-frame uniform — ModelPool.update writes the camera's
    // projection*view matrix into here so the vertex shader can do GPU
    // frustum culling without a CPU sphere test per entity.
    this._uniforms = { projViewMatrix: { value: new THREE.Matrix4() } };
    _patchInstancedSlotMaterial(material, this._uniforms);
    this.mesh = new THREE.InstancedMesh(geo, material, this.capacity);
    this.mesh.frustumCulled = false; // GPU vertex-shader handles culling
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    // Per-instance world-space bounding sphere (cx, cy, cz, r). Set on
    // slot acquire / update; the vertex shader reads this and collapses
    // out-of-frustum instances to NaN.
    this._boundArray = new Float32Array(this.capacity * 4);
    this._boundAttr = new THREE.InstancedBufferAttribute(this._boundArray, 4);
    this._boundAttr.setUsage(THREE.DynamicDrawUsage);
    this.mesh.geometry.setAttribute('instanceBoundSphere', this._boundAttr);
    // Zero out all instance matrices initially so unused slots draw nothing
    // visible (zero matrix collapses to origin point).
    const zero = new THREE.Matrix4().set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
    for (let i = 0; i < this.capacity; i++) this.mesh.setMatrixAt(i, zero);
    this.mesh.count = 0;
    this.mesh.instanceMatrix.needsUpdate = true;
    this.slots = new Map(); // entity -> slot index
    this.freeSlots = []; // recycled indices
    this.nextSlot = 0;
  }

  acquireSlot(entity) {
    let idx;
    if (this.freeSlots.length) idx = this.freeSlots.pop();
    else {
      if (this.nextSlot >= this.capacity) this._grow(this.capacity * 2);
      idx = this.nextSlot++;
    }
    this.slots.set(entity, idx);
    if (idx + 1 > this.mesh.count) this.mesh.count = idx + 1;
    return idx;
  }
  releaseSlot(entity) {
    const idx = this.slots.get(entity);
    if (idx == null) return;
    this.slots.delete(entity);
    this.freeSlots.push(idx);
    // Zero its matrix so it stops drawing.
    const zero = new THREE.Matrix4().set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
    this.mesh.setMatrixAt(idx, zero);
    this.mesh.instanceMatrix.needsUpdate = true;
    // Zero the bound-sphere radius so the shader treats this slot as
    // "no bound info" → also drawn at origin (zero matrix). Belt+braces.
    const o = idx * 4;
    this._boundArray[o] = 0; this._boundArray[o+1] = 0; this._boundArray[o+2] = 0; this._boundArray[o+3] = 0;
    this._boundAttr.needsUpdate = true;
  }
  setMatrixForSlot(idx, matrix) {
    this.mesh.setMatrixAt(idx, matrix);
    this.mesh.instanceMatrix.needsUpdate = true;
  }
  setBoundSphereForSlot(idx, cx, cy, cz, r) {
    const o = idx * 4;
    this._boundArray[o] = cx;
    this._boundArray[o+1] = cy;
    this._boundArray[o+2] = cz;
    this._boundArray[o+3] = r;
    this._boundAttr.needsUpdate = true;
  }
  _grow(newCap) {
    const old = this.mesh;
    const next = new THREE.InstancedMesh(this.geometry, this.material, newCap);
    next.frustumCulled = false;
    next.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const m = new THREE.Matrix4();
    for (let i = 0; i < this.nextSlot; i++) {
      old.getMatrixAt(i, m);
      next.setMatrixAt(i, m);
    }
    next.count = old.count;
    next.instanceMatrix.needsUpdate = true;
    // Grow + carry the per-instance bound-sphere attribute.
    const newBounds = new Float32Array(newCap * 4);
    newBounds.set(this._boundArray);
    this._boundArray = newBounds;
    this._boundAttr = new THREE.InstancedBufferAttribute(newBounds, 4);
    this._boundAttr.setUsage(THREE.DynamicDrawUsage);
    next.geometry.setAttribute('instanceBoundSphere', this._boundAttr);
    const parent = old.parent;
    if (parent) {
      parent.remove(old);
      parent.add(next);
    }
    old.dispose();
    this.mesh = next;
    this.capacity = newCap;
  }
}

// Patch a material so its vertex shader receives a per-instance bound-sphere
// attribute and a per-frame projViewMatrix uniform, then collapses any
// instance outside the camera frustum to a NaN clip-space position so the GPU
// early-rejects it. Wraps any existing onBeforeCompile so the vertex-color
// gamma patch on the fragment side still runs.
function _patchInstancedSlotMaterial(material, uniforms) {
  const prev = material.onBeforeCompile;
  material.onBeforeCompile = (shader) => {
    if (prev) prev(shader);
    shader.uniforms.projViewMatrix = uniforms.projViewMatrix;
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
attribute vec4 instanceBoundSphere;
uniform mat4 projViewMatrix;`
      )
      .replace(
        '#include <project_vertex>',
        `#include <project_vertex>
{
  // GPU frustum cull: derive 6 clip-space planes from projViewMatrix and
  // test the world-space bounding sphere against each. If fully outside any
  // plane, collapse the vertex to NaN so the GPU early-rejects this triangle
  // before rasterization. Multiplying NaN by 0 keeps NaN, so even drivers
  // that flake on NaN compares still discard it.
  if (instanceBoundSphere.w > 0.0) {
    vec3 c = instanceBoundSphere.xyz;
    float r = instanceBoundSphere.w;
    // rows of projViewMatrix (column-major storage: M[col][row])
    vec4 row0 = vec4(projViewMatrix[0][0], projViewMatrix[1][0], projViewMatrix[2][0], projViewMatrix[3][0]);
    vec4 row1 = vec4(projViewMatrix[0][1], projViewMatrix[1][1], projViewMatrix[2][1], projViewMatrix[3][1]);
    vec4 row2 = vec4(projViewMatrix[0][2], projViewMatrix[1][2], projViewMatrix[2][2], projViewMatrix[3][2]);
    vec4 row3 = vec4(projViewMatrix[0][3], projViewMatrix[1][3], projViewMatrix[2][3], projViewMatrix[3][3]);
    vec4 planes[6];
    planes[0] = row3 + row0; // left
    planes[1] = row3 - row0; // right
    planes[2] = row3 + row1; // bottom
    planes[3] = row3 - row1; // top
    planes[4] = row3 + row2; // near
    planes[5] = row3 - row2; // far
    bool outside = false;
    for (int i = 0; i < 6; i++) {
      vec4 p = planes[i];
      float len = length(p.xyz);
      if (len > 0.0) {
        float d = (dot(p.xyz, c) + p.w) / len;
        if (d < -r) { outside = true; break; }
      }
    }
    if (outside) {
      gl_Position = vec4(0.0/0.0, 0.0/0.0, 0.0/0.0, 0.0/0.0) * 0.0;
      return;
    }
  }
}`
      );
  };
  material.needsUpdate = true;
}

// --- Asset: shared resources for one source URL ---------------------------
// Loaded once, referenced by N Entity instances.
class Asset {
  constructor(pool, url) {
    this.pool = pool;
    this.url = url;
    this.state = 'pending'; // 'pending' | 'loading' | 'ready' | 'error'
    this.error = null;
    // baseDir is the directory of the root model.progressive.glb so we can
    // resolve sibling LOD relative paths against it.
    this.baseDir = url.endsWith('/') ? url : url.replace(/[^/]+$/, '');
    // Per-mesh LOD descriptors from the LOCAL_progressive extras blob, sorted
    // by quality ascending (idx 0 = lowest, idx N-1 = highest).
    this.meshLodDescs = []; // [{ meshIndex, primIndex, lods: [...] }]
    this.texLodDescs = []; // [{ textureIndex, name, lods: [...] }]
    // Cached shared geometries: key `${meshIndex}:${primIndex}:${lodIdx}` -> BufferGeometry
    this.geoCache = new Map();
    // Cached shared texture bitmaps: key `${textureIndex}:${lodIdx}` -> ImageBitmap
    this.texCache = new Map();
    // LRU touch timestamps (performance.now() at last cache hit/fill).
    // Eviction in _enforceBudget walks these ASC for working-set retention.
    this.geoTouch = new Map(); // same key as geoCache
    this.texTouch = new Map(); // same key as texCache
    // The original gltf payload from the root load — used to clone scenes
    // per-entity. Held as the parsed three.js Object3D plus parser.json.
    this.rootGltf = null;
    // VRM extension blob (if present) so spawned entities can re-bind to a
    // matching three-vrm runtime per entity.
    this.hasVRM = false;
    // Track bytes-loaded per LOD for the budget system in Phase C.
    this.byteWeights = new Map(); // key -> bytes
    // Loaders need to know whether this asset is VRM-bearing (root) or a
    // plain sibling LOD (no VRM).
    this._rootLoader = _makeLoader(true);
    this._lodLoader = _makeLoader(false);
    // Promise that resolves when the root is parsed.
    this.ready = this._load();
  }

  async _fetchBytes(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
    const buf = new Uint8Array(await res.arrayBuffer());
    this.pool._trackBytes(this.url, url, buf.byteLength);
    return buf;
  }

  // HTTP byte-range fetch — used by the streaming-GLB path to pull just the
  // BIN slice owning a single LOD's bufferViews. byteOffset here is
  // RELATIVE to the BIN chunk; we add _binBase to get an absolute file
  // offset for the Range header.
  async _fetchRange(url, byteOffset, byteLength) {
    const absStart = (this._binBase || 0) + byteOffset;
    const absEnd = absStart + byteLength - 1;
    const res = await fetch(url, { headers: { Range: `bytes=${absStart}-${absEnd}` } });
    // 206 Partial Content is the expected status; some dev servers may
    // return 200 with the full file if Range isn't honored — accept both.
    if (!res.ok && res.status !== 206) throw new Error(`fetch ${url} range ${absStart}-${absEnd}: ${res.status}`);
    const buf = new Uint8Array(await res.arrayBuffer());
    if (res.status === 200 && buf.byteLength > byteLength) {
      // Server ignored Range — slice client-side using absolute offset.
      return buf.subarray(absStart, absStart + byteLength);
    }
    return buf;
  }

  // Build a minimal-glTF JSON spec for ONE mesh LOD: a single mesh+primitive
  // referencing the LOD's accessors (POSITION, NORMAL, INDICES, ...), with
  // bufferViews remapped to byteOffsets relative to a freshly-zeroed BIN
  // buffer whose total length === target.byteLength.
  _buildStreamingLodSpec(target) {
    const accs = this._streamAccessors;
    const bvs = this._streamBufferViews;
    if (!accs || !bvs) throw new Error('streaming: missing accessor/bufferView tables');
    const attrAccs = target.attrAccs || {};
    // Collect every accessor index used by this LOD (indices + attributes).
    const accIndices = new Set();
    if (target.indicesAcc != null) accIndices.add(target.indicesAcc);
    for (const k of Object.keys(attrAccs)) accIndices.add(attrAccs[k]);
    // Each used accessor refers to a bufferView; collect those.
    const bvOldToNew = new Map();
    const newBvs = [];
    for (const ai of accIndices) {
      const a = accs[ai];
      if (!a || a.bufferView == null) continue;
      if (bvOldToNew.has(a.bufferView)) continue;
      const bv = bvs[a.bufferView];
      if (!bv) continue;
      // The contiguous range [target.byteOffset .. +byteLength) holds ALL
      // these bufferViews. Remap each bv's byteOffset to be relative to
      // target.byteOffset, into a new single-buffer glTF.
      const newOffset = bv.byteOffset - target.byteOffset;
      if (newOffset < 0 || newOffset + bv.byteLength > target.byteLength) {
        throw new Error(`streaming: bufferView ${a.bufferView} (offset ${bv.byteOffset}, len ${bv.byteLength}) out of LOD range [${target.byteOffset}, ${target.byteOffset + target.byteLength})`);
      }
      const nb = { buffer: 0, byteOffset: newOffset, byteLength: bv.byteLength };
      if (bv.byteStride != null) nb.byteStride = bv.byteStride;
      if (bv.target != null) nb.target = bv.target;
      bvOldToNew.set(a.bufferView, newBvs.length);
      newBvs.push(nb);
    }
    // Build new accessor list, in a deterministic order, capturing old->new
    // index map so primitive references stay valid.
    const accOldToNew = new Map();
    const newAccs = [];
    for (const ai of accIndices) {
      const a = accs[ai];
      if (!a) continue;
      const newBv = bvOldToNew.get(a.bufferView);
      if (newBv == null) continue;
      const nA = {
        bufferView: newBv,
        byteOffset: a.byteOffset,
        componentType: a.componentType,
        count: a.count,
        type: a.type,
      };
      if (a.normalized) nA.normalized = true;
      if (a.min) nA.min = a.min;
      if (a.max) nA.max = a.max;
      accOldToNew.set(ai, newAccs.length);
      newAccs.push(nA);
    }
    const primitive = { attributes: {} };
    for (const k of Object.keys(attrAccs)) {
      const mapped = accOldToNew.get(attrAccs[k]);
      if (mapped != null) primitive.attributes[k] = mapped;
    }
    if (target.indicesAcc != null) {
      const mapped = accOldToNew.get(target.indicesAcc);
      if (mapped != null) primitive.indices = mapped;
    }
    const spec = {
      asset: { version: '2.0', generator: 'gltf-progressive streaming sub-GLB' },
      buffers: [{ byteLength: target.byteLength }],
      bufferViews: newBvs,
      accessors: newAccs,
      meshes: [{ primitives: [primitive] }],
      nodes: [{ mesh: 0 }],
      scenes: [{ nodes: [0] }],
      scene: 0,
    };
    if (this._streamExtensionsUsed) spec.extensionsUsed = this._streamExtensionsUsed;
    if (this._streamExtensionsRequired) spec.extensionsRequired = this._streamExtensionsRequired;
    return spec;
  }

  async _load() {
    this.state = 'loading';
    try {
      const rootBytes = await this._fetchBytes(this.url);
      const gltf = await new Promise((resolve, reject) => {
        this._rootLoader.parse(rootBytes.buffer, '', resolve, reject);
      });
      this.rootGltf = gltf;
      this.hasVRM = !!gltf.userData?.vrm;
      const ext = gltf.parser.json?.extras?.LOCAL_progressive;
      // Streaming mode: a single .glb hosts every LOD's bufferViews inside
      // its BIN chunk. Sibling fetches are replaced by HTTP Range requests
      // against `this.url`. We need the JSON descriptor in scope to look up
      // accessor + bufferView records when building the per-LOD sub-GLB.
      this.streamingMode = !!(ext && ext.streaming === true && ext.version >= 2);
      if (this.streamingMode) {
        const j = gltf.parser.json || {};
        this._gltfJson = j;
        // Compute the absolute file-offset where the BIN chunk's payload
        // starts. extras.LOCAL_progressive byteOffsets are RELATIVE to the
        // BIN chunk; we add this base to form absolute Range requests.
        // GLB layout: 12-byte header + 8-byte JSON chunk header + JSON +
        // 8-byte BIN chunk header + BIN bytes. JSON length lives at offset
        // 12 (uint32 LE).
        {
          const dv = new DataView(rootBytes.buffer, rootBytes.byteOffset, rootBytes.byteLength);
          const jsonChunkLen = dv.getUint32(12, true);
          this._binBase = 12 + 8 + jsonChunkLen + 8;
        }
        // Snapshot just what we need; the parser's `json` may otherwise be
        // mutated by extension preprocessors. Bufferviews carry byteOffsets
        // we'll remap; accessors carry component/count/type info.
        this._streamBufferViews = (j.bufferViews || []).map((bv) => ({
          buffer: bv.buffer,
          byteOffset: bv.byteOffset || 0,
          byteLength: bv.byteLength,
          byteStride: bv.byteStride,
          target: bv.target,
        }));
        this._streamAccessors = (j.accessors || []).map((a) => ({
          bufferView: a.bufferView,
          byteOffset: a.byteOffset || 0,
          componentType: a.componentType,
          normalized: !!a.normalized,
          count: a.count,
          type: a.type,
          min: a.min,
          max: a.max,
          sparse: a.sparse,
        }));
        // KHR_mesh_quantization may be needed when the meshopt'd LODs use
        // quantized attribute component types — preserve original extension
        // declarations.
        this._streamExtensionsUsed = j.extensionsUsed ? [...j.extensionsUsed] : undefined;
        this._streamExtensionsRequired = j.extensionsRequired ? [...j.extensionsRequired] : undefined;
      }
      if (ext) {
        const kindRank = { unskinned: 0, vertcolor: 1, textured: 2 };
        for (const m of ext.meshes) {
          const sorted = [...m.lods].sort((a, b) => {
            const ra = kindRank[a.kind || 'textured'] ?? 2;
            const rb = kindRank[b.kind || 'textured'] ?? 2;
            if (ra !== rb) return ra - rb;
            return (a.ratio || 0) - (b.ratio || 0);
          });
          this.meshLodDescs.push({ meshIndex: m.meshIndex, primIndex: m.primIndex, lods: sorted });
        }
        for (const t of ext.textures) {
          const sortedT = [...t.lods].sort((a, b) => a.width - b.width);
          this.texLodDescs.push({ textureIndex: t.textureIndex, name: t.name, lods: sortedT });
        }
      }
      // Pre-cache the inline (lowest-textured) geometry per primitive AND the
      // smallest textures from the root — they're already in the parsed
      // gltf scene; no second fetch needed.
      let meshIdx = 0;
      gltf.scene.traverse((c) => {
        if (c.isMesh) {
          const desc = this.meshLodDescs[meshIdx];
          if (desc) {
            const inlineLodIdx = desc.lods.findIndex((l) => l.inline);
            if (inlineLodIdx >= 0) {
              this.geoCache.set(`${desc.meshIndex}:${desc.primIndex}:${inlineLodIdx}`, c.geometry);
              this.geoTouch.set(`${desc.meshIndex}:${desc.primIndex}:${inlineLodIdx}`, _now());
            }
          }
          meshIdx++;
        }
      });
      // Cache the inline-sized texture bitmaps from the parsed scene.
      gltf.scene.traverse((c) => {
        if (!c.isMesh || !c.material) return;
        const slots = ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'emissiveMap'];
        for (const s of slots) {
          const tex = c.material[s];
          if (!tex || !tex.image) continue;
          // Find which descriptor this texture belongs to (name match).
          const desc = this.texLodDescs.find((d) => d.name === tex.name);
          if (desc) {
            const inlineIdx = desc.lods.findIndex((l) => l.inline);
            if (inlineIdx >= 0) {
              this.texCache.set(`${desc.textureIndex}:${inlineIdx}`, tex.image);
              this.texTouch.set(`${desc.textureIndex}:${inlineIdx}`, _now());
            }
          }
        }
      });
      this.state = 'ready';
    } catch (e) {
      this.state = 'error';
      this.error = e;
      throw e;
    }
  }

  // Fetch a mesh LOD's shared geometry. Returns a Promise<BufferGeometry>.
  // Triggers the pool's per-asset request queue.
  async ensureMeshLod(meshDescIdx, lodIdx) {
    const desc = this.meshLodDescs[meshDescIdx];
    if (!desc) return null;
    const target = desc.lods[lodIdx];
    if (!target) return null;
    const key = `${desc.meshIndex}:${desc.primIndex}:${lodIdx}`;
    const cached = this.geoCache.get(key);
    if (cached) { this.geoTouch.set(key, _now()); return cached; }
    if (target.inline) return null; // should already be cached from root load
    // De-dupe in-flight requests through the pool's load queue (Phase C).
    return this.pool._enqueue(`${this.url}#${key}`, async () => {
      const stillCached = this.geoCache.get(key);
      if (stillCached) { this.geoTouch.set(key, _now()); return stillCached; }
      // ----- Streaming-GLB path ---------------------------------------------
      // No sibling file: same `this.url`, fetched with a Range header that
      // covers exactly the bufferViews this LOD owns. We synthesize a tiny
      // standalone GLB whose JSON references those bufferViews remapped to a
      // zero-based BIN buffer, then hand THAT to the worker (or main thread)
      // which runs the existing GLTFLoader.parse + _bakeQuantizeDecode pipe.
      if (this.streamingMode && (target.byteOffset != null) && (target.byteLength != null)) {
        const spec = this._buildStreamingLodSpec(target);
        if (this.pool._workers.length) {
          try {
            const payload = await this.pool._workerFetchStreamingLod(
              this.url, target.byteOffset, target.byteLength, spec, target.decodeAABB, this._binBase || 0,
            );
            this.pool._trackBytes(this.url, this.url, payload.bytes);
            const geo = ModelPool._buildGeometryFromPayload(payload);
            this.geoCache.set(key, geo);
            this.geoTouch.set(key, _now());
            this.byteWeights.set(key, payload.bytes);
            return geo;
          } catch (e) {
            console.warn('[asset] streaming worker decode failed, fallback main thread', e);
          }
        }
        // Main-thread streaming fallback.
        const slice = await this._fetchRange(this.url, target.byteOffset, target.byteLength);
        const glb = _composeSubGLB(spec, slice);
        const gltf = await new Promise((resolve, reject) => {
          this._lodLoader.parse(glb.buffer, '', resolve, reject);
        });
        let srcMesh = null;
        gltf.scene.updateMatrixWorld(true);
        gltf.scene.traverse((c) => { if (c.isMesh && !srcMesh) srcMesh = c; });
        const geo = srcMesh?.geometry;
        if (geo) {
          _bakeQuantizeDecode(geo, srcMesh.matrixWorld, target.decodeAABB);
          this.geoCache.set(key, geo);
          this.geoTouch.set(key, _now());
          this.byteWeights.set(key, slice.byteLength);
        }
        return geo;
      }
      // ----- Sibling-file path (back-compat) --------------------------------
      const fullUrl = this.baseDir + target.path;
      // Worker path: fetch + parse + bake-decode all happen off-thread; we
      // get back a payload of transferable typed arrays and a small bbox
      // record. Main thread only allocates a BufferGeometry shell wiring
      // those arrays as attributes — no per-vertex JS loops.
      if (this.pool._workers.length) {
        try {
          const payload = await this.pool._workerFetchLod(fullUrl, target.decodeAABB);
          this.pool._trackBytes(this.url, fullUrl, payload.bytes);
          const geo = ModelPool._buildGeometryFromPayload(payload);
          this.geoCache.set(key, geo);
          this.geoTouch.set(key, _now());
          this.byteWeights.set(key, payload.bytes);
          return geo;
        } catch (e) {
          // Fall through to main-thread path on worker failure.
          console.warn('[asset] worker decode failed, fallback main thread', e);
        }
      }
      // Main-thread fallback (or workerCount: 0).
      const bytes = await this._fetchBytes(fullUrl);
      const gltf = await new Promise((resolve, reject) => {
        this._lodLoader.parse(bytes.buffer, '', resolve, reject);
      });
      let srcMesh = null;
      gltf.scene.updateMatrixWorld(true);
      gltf.scene.traverse((c) => { if (c.isMesh && !srcMesh) srcMesh = c; });
      const geo = srcMesh?.geometry;
      if (geo) {
        _bakeQuantizeDecode(geo, srcMesh.matrixWorld, target.decodeAABB);
        this.geoCache.set(key, geo);
        this.geoTouch.set(key, _now());
        this.byteWeights.set(key, bytes.byteLength);
      }
      return geo;
    });
  }

  async ensureTexLod(texDescIdx, lodIdx) {
    const desc = this.texLodDescs[texDescIdx];
    if (!desc) return null;
    const target = desc.lods[lodIdx];
    if (!target) return null;
    const key = `${desc.textureIndex}:${lodIdx}`;
    const cached = this.texCache.get(key);
    if (cached) { this.texTouch.set(key, _now()); return cached; }
    if (target.inline) return null;
    return this.pool._enqueue(`${this.url}#tex:${key}`, async () => {
      const stillCached = this.texCache.get(key);
      if (stillCached) { this.texTouch.set(key, _now()); return stillCached; }
      let bytes;
      if (this.streamingMode && target.byteOffset != null && target.byteLength != null) {
        bytes = await this._fetchRange(this.url, target.byteOffset, target.byteLength);
        this.pool._trackBytes(this.url, `${this.url}#tex:${key}`, bytes.byteLength);
      } else {
        bytes = await this._fetchBytes(this.baseDir + target.path);
      }
      const blob = new Blob([bytes], { type: target.mime || 'image/webp' });
      const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none' });
      this.texCache.set(key, bmp);
      this.texTouch.set(key, _now());
      this.byteWeights.set(`tex:${key}`, bytes.byteLength);
      return bmp;
    });
  }

  // Evict a LOD's cached resource (Phase C). Called by the pool.
  evictMeshLod(meshDescIdx, lodIdx) {
    const desc = this.meshLodDescs[meshDescIdx];
    if (!desc) return false;
    const key = `${desc.meshIndex}:${desc.primIndex}:${lodIdx}`;
    const target = desc.lods[lodIdx];
    if (target?.inline) return false; // never evict inline geometry
    const geo = this.geoCache.get(key);
    if (!geo) return false;
    geo.dispose();
    this.geoCache.delete(key);
    this.geoTouch.delete(key);
    this.byteWeights.delete(key);
    return true;
  }
  evictTexLod(texDescIdx, lodIdx) {
    const desc = this.texLodDescs[texDescIdx];
    if (!desc) return false;
    const key = `${desc.textureIndex}:${lodIdx}`;
    const target = desc.lods[lodIdx];
    if (target?.inline) return false;
    const bmp = this.texCache.get(key);
    if (!bmp) return false;
    if (bmp.close) bmp.close();
    this.texCache.delete(key);
    this.texTouch.delete(key);
    this.byteWeights.delete(`tex:${key}`);
    return true;
  }
}

// Repack interleaved attributes into standalone Float32 and bake the source
// mesh's local matrix (which carries the dequantize transform for plain
// Meshes loaded via GLTFLoader+KHR_mesh_quantization) into vertex data.
// When the matrix is identity, fall back to scanning the actual post-
// dequantize range and remapping into the per-LOD decodeAABB captured at
// bake time. This is the same logic the inline demo used; lifted into a
// helper so the pool and any direct consumers can share it.
function _bakeQuantizeDecode(geo, matrix, decodeAABB) {
  const m = matrix;
  const isIdentity = (
    m.elements[0] === 1 && m.elements[5] === 1 && m.elements[10] === 1 &&
    m.elements[12] === 0 && m.elements[13] === 0 && m.elements[14] === 0 &&
    m.elements[1] === 0 && m.elements[2] === 0 && m.elements[4] === 0 &&
    m.elements[6] === 0 && m.elements[8] === 0 && m.elements[9] === 0
  );
  if (!isIdentity) {
    for (const semKey of ['position', 'normal', 'tangent']) {
      const a = geo.attributes[semKey];
      if (!a) continue;
      const out = new Float32Array(a.count * a.itemSize);
      for (let i = 0; i < a.count; i++) {
        if (a.itemSize >= 1) out[i * a.itemSize + 0] = a.getX(i);
        if (a.itemSize >= 2) out[i * a.itemSize + 1] = a.getY(i);
        if (a.itemSize >= 3) out[i * a.itemSize + 2] = a.getZ(i);
        if (a.itemSize >= 4) out[i * a.itemSize + 3] = a.getW(i);
      }
      geo.setAttribute(semKey, new THREE.BufferAttribute(out, a.itemSize, false));
    }
    geo.applyMatrix4(m);
  } else if (decodeAABB) {
    const { min, max } = decodeAABB;
    const pos = geo.attributes.position;
    if (pos) {
      let smnX = Infinity, smxX = -Infinity;
      let smnY = Infinity, smxY = -Infinity;
      let smnZ = Infinity, smxZ = -Infinity;
      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
        if (x < smnX) smnX = x; if (x > smxX) smxX = x;
        if (y < smnY) smnY = y; if (y > smxY) smxY = y;
        if (z < smnZ) smnZ = z; if (z > smxZ) smxZ = z;
      }
      const r = (a, b) => (b - a < 1e-9 ? 1 : b - a);
      const sx = (max[0] - min[0]) / r(smnX, smxX);
      const sy = (max[1] - min[1]) / r(smnY, smxY);
      const sz = (max[2] - min[2]) / r(smnZ, smxZ);
      const out = new Float32Array(pos.count * 3);
      for (let i = 0; i < pos.count; i++) {
        out[i * 3 + 0] = (pos.getX(i) - smnX) * sx + min[0];
        out[i * 3 + 1] = (pos.getY(i) - smnY) * sy + min[1];
        out[i * 3 + 2] = (pos.getZ(i) - smnZ) * sz + min[2];
      }
      geo.setAttribute('position', new THREE.BufferAttribute(out, 3, false));
      for (const semKey of ['normal', 'tangent']) {
        const a = geo.attributes[semKey];
        if (!a) continue;
        const o = new Float32Array(a.count * a.itemSize);
        for (let i = 0; i < a.count; i++) {
          if (a.itemSize >= 1) o[i * a.itemSize + 0] = a.getX(i);
          if (a.itemSize >= 2) o[i * a.itemSize + 1] = a.getY(i);
          if (a.itemSize >= 3) o[i * a.itemSize + 2] = a.getZ(i);
          if (a.itemSize >= 4) o[i * a.itemSize + 3] = a.getW(i);
        }
        geo.setAttribute(semKey, new THREE.BufferAttribute(o, a.itemSize, false));
      }
    }
  }
  geo.computeBoundingSphere();
  geo.computeBoundingBox();
}

// --- Entity: one live instance --------------------------------------------
// Encapsulates the per-instance THREE.Object3D tree (root Object3D containing
// SkinnedMesh / Mesh + skeleton clone), per-mesh LOD state, and per-frame
// update logic.
class Entity extends Emitter {
  constructor(pool, asset, opts) {
    super();
    this.pool = pool;
    this.asset = asset;
    this.id = ++pool._nextEntityId;
    this.opts = opts || {};
    // Root container — application code can `.add()` it to a scene, set
    // position/rotation/scale on it, etc.
    this.root = new THREE.Object3D();
    this.root.name = `entity_${this.id}_${asset.url.split('/').pop()}`;
    if (opts.position) this.root.position.fromArray(opts.position);
    if (opts.rotation) this.root.quaternion.setFromEuler(new THREE.Euler().fromArray(opts.rotation));
    if (opts.scale) this.root.scale.setScalar(opts.scale);
    // Caller can pass `static: true` for entities that never move after
    // spawn — we then disable auto matrix updates after composing position
    // into matrix once. Subsequent frames skip the matrix recompute walk.
    // Critical: matrixAutoUpdate=false means three.js's render-time
    // updateMatrixWorld skips updateMatrix(), so we must call it manually
    // here while position/quaternion/scale are still being read into matrix.
    if (opts.static) {
      this.root.updateMatrix(); // compose position/quat/scale into matrix
      this.root.matrixAutoUpdate = false;
      this.root.matrixWorldNeedsUpdate = true; // force one world recompute
    }
    // Per-mesh tracking: each tracks current LOD, the live SkinnedMesh/Mesh,
    // and per-tex current LOD.
    this.trackedMeshes = []; // [{ meshDescIdx, currentLod, mesh, texState: [{currentLod}, ...], baseSkeleton, baseMaterial, sharedTextures }]
    // Animation state.
    this.animationMixer = null;
    this.animationClips = [];
    this.animationAction = null;
    // VRM runtime (shared across LODs of one entity).
    this.vrm = null;
    // Frustum culling cached state.
    this._lastInFrustum = true;
    // Disposed flag — stop touching this entity after dispose().
    this._disposed = false;
    // Scene-parent tracking: when ALL tracked meshes are routed through
    // InstancedMesh slots, we detach `root` from its scene parent so three.js
    // stops walking it during updateMatrixWorld/render-list construction. The
    // instanced matrix in the per-asset InstancedMesh is the only state the
    // renderer needs. We remember the parent so we can re-attach when the
    // entity gets closer and needs its per-entity SkinnedMesh tree again.
    this._sceneParent = null;
    this._detached = false;
    // Ready promise resolves once the first LOD has been applied and the
    // root contains a renderable mesh.
    this.ready = this._bootstrap();
  }

  async _bootstrap() {
    try {
      await this.asset.ready;
      if (this._disposed) return;
      // Clone the root gltf scene per-entity so each has its own skeleton +
      // mesh objects (geometries are shared via the asset cache).
      const sourceScene = this.asset.rootGltf.scene;
      const cloned = _cloneSkinned(sourceScene);
      // Find the cloned VRM if present.
      this.vrm = this.asset.rootGltf.userData?.vrm || null;
      this.root.add(cloned);
      // Discover tracked meshes by descriptor.
      const meshOrder = [];
      cloned.traverse((c) => { if (c.isMesh) meshOrder.push(c); });
      for (let i = 0; i < this.asset.meshLodDescs.length; i++) {
        const desc = this.asset.meshLodDescs[i];
        const mesh = meshOrder[i] || meshOrder[0];
        if (!mesh) continue;
        const inlineLodIdx = desc.lods.findIndex((l) => l.inline);
        this.trackedMeshes.push({
          meshDescIdx: i,
          currentLod: inlineLodIdx >= 0 ? inlineLodIdx : 0,
          mesh,
          baseIsSkinnedMesh: !!mesh.isSkinnedMesh,
          baseMaterial: mesh.material,
          baseSkeleton: mesh.skeleton || null,
          parent: mesh.parent,
          texState: this.asset.texLodDescs.map(() => ({ currentLod: 0 })),
          vcMaterial: null,
        });
      }
      // Animation: build mixer if source has clips.
      const animations = this.asset.rootGltf.animations || [];
      if (animations.length) {
        this.animationClips = animations;
        this.animationMixer = new THREE.AnimationMixer(cloned);
        const desiredIdx = Math.min(this.opts.animationIndex ?? 0, animations.length - 1);
        this.animationAction = this.animationMixer.clipAction(animations[desiredIdx]);
        this.animationAction.setLoop(THREE.LoopRepeat).play();
      }
      this.emit('ready', this);
    } catch (e) {
      this.emit('error', e);
    }
  }

  // Update one tracked mesh to the desired LOD index. Materializes the
  // shell-type change (SkinnedMesh ↔ Mesh) and material swap as needed.
  async _applyLod(tm, wantIdx) {
    if (this._disposed) return;
    if (wantIdx === tm.currentLod) return;
    const desc = this.asset.meshLodDescs[tm.meshDescIdx];
    if (!desc) return;
    const target = desc.lods[wantIdx];
    if (!target) return;
    // Fetch geometry (cached or on-demand).
    let geo;
    if (target.inline) {
      const _gk = `${desc.meshIndex}:${desc.primIndex}:${wantIdx}`;
      geo = this.asset.geoCache.get(_gk);
      if (geo) this.asset.geoTouch.set(_gk, _now());
    } else {
      geo = await this.asset.ensureMeshLod(tm.meshDescIdx, wantIdx);
    }
    if (this._disposed || !geo) return;
    if (wantIdx === tm.currentLod) return; // raced
    const kind = target.kind || 'textured';
    const wantSkinned = kind !== 'unskinned' && tm.baseIsSkinnedMesh;
    const haveSkinned = !!tm.mesh.isSkinnedMesh;

    // Instanced mode: when the target LOD is unskinned, route this entity
    // through a shared InstancedMesh slot instead of keeping its own mesh
    // tree. This collapses N entities into 1 draw call at the lowest LOD.
    const wantInstanced = kind === 'unskinned';
    const haveInstanced = tm._instancedSlot != null;
    if (wantInstanced) {
      const slot = this.pool._getInstancedSlot(this.asset, tm.meshDescIdx, wantIdx);
      if (slot) {
        // Hide the entity's own mesh from the renderer.
        tm.mesh.visible = false;
        // Acquire a slot index (re-acquire if changing LOD within unskinned tier).
        if (haveInstanced && (tm._instancedSlot !== slot)) {
          tm._instancedSlot.releaseSlot(this);
        }
        if (!haveInstanced || tm._instancedSlot !== slot) {
          tm._instancedSlot = slot;
          tm._instancedSlotIdx = slot.acquireSlot(this);
        }
        // Seed the per-instance world-space bound sphere for GPU culling.
        // Recomputed once here (entity transform is stable for typical
        // static spawns; movers refresh it in _update). For animated
        // entities this is fine — far-tier instanced LODs are unskinned
        // bind-pose, so the sphere envelope is constant.
        {
          const sphere = geo.boundingSphere;
          if (sphere) {
            this.root.updateMatrixWorld(true);
            const scale = this.root.scale.length() / Math.SQRT2;
            const cx = this.root.matrixWorld.elements[12];
            const cy = this.root.matrixWorld.elements[13];
            const cz = this.root.matrixWorld.elements[14];
            const r = sphere.radius * scale;
            tm._instancedBoundRadius = sphere.radius;
            slot.setBoundSphereForSlot(tm._instancedSlotIdx, cx, cy, cz, r);
          }
        }
        tm.currentLod = wantIdx;
        this.emit('lod-changed', { entity: this, meshDescIdx: tm.meshDescIdx, lod: wantIdx, kind, instanced: true });
        return;
      }
      // Slot unavailable (geo not yet loaded for instancing) → fall through to non-instanced path.
    } else if (haveInstanced) {
      // Leaving the instanced tier — release the slot and re-show our own mesh.
      tm._instancedSlot.releaseSlot(this);
      tm._instancedSlot = null;
      tm._instancedSlotIdx = -1;
      tm.mesh.visible = true;
    }
    // Material selection per kind.
    let mat;
    if (kind === 'textured') {
      mat = tm.baseMaterial;
    } else {
      if (!tm.vcMaterial) {
        const m = new THREE.MeshLambertMaterial({ vertexColors: true });
        m.onBeforeCompile = (shader) => {
          shader.fragmentShader = shader.fragmentShader.replace(
            '#include <color_fragment>',
            `#if defined( USE_COLOR_ALPHA )
              diffuseColor.rgb *= pow(vColor.rgb, vec3(2.2));
              diffuseColor.a *= vColor.a;
            #elif defined( USE_COLOR )
              diffuseColor.rgb *= pow(vColor, vec3(2.2));
            #endif`
          );
        };
        tm.vcMaterial = m;
      }
      mat = tm.vcMaterial;
    }
    if (wantSkinned === haveSkinned) {
      tm.mesh.geometry = geo;
      tm.mesh.material = mat;
    } else {
      const parent = tm.mesh.parent || tm.parent;
      let next;
      if (wantSkinned) {
        next = new THREE.SkinnedMesh(geo, mat);
        if (tm.baseSkeleton) next.bind(tm.baseSkeleton);
      } else {
        next = new THREE.Mesh(geo, mat);
      }
      next.frustumCulled = false;
      next.position.copy(tm.mesh.position);
      next.quaternion.copy(tm.mesh.quaternion);
      next.scale.copy(tm.mesh.scale);
      next.name = tm.mesh.name;
      if (parent) {
        parent.remove(tm.mesh);
        parent.add(next);
      }
      tm.mesh = next;
    }
    tm.currentLod = wantIdx;
    this.emit('lod-changed', { entity: this, meshDescIdx: tm.meshDescIdx, lod: wantIdx, kind });
  }

  async _applyTexLod(tm, tdIdx, wantIdx) {
    if (this._disposed) return;
    const tState = tm.texState[tdIdx];
    if (!tState || wantIdx === tState.currentLod) return;
    const desc = this.asset.texLodDescs[tdIdx];
    if (!desc) return;
    let bmp;
    const target = desc.lods[wantIdx];
    if (target.inline) {
      const _tk = `${desc.textureIndex}:${wantIdx}`;
      bmp = this.asset.texCache.get(_tk);
      if (bmp) this.asset.texTouch.set(_tk, _now());
    } else {
      bmp = await this.asset.ensureTexLod(tdIdx, wantIdx);
    }
    if (this._disposed || !bmp) return;
    if (wantIdx === tState.currentLod) return;
    // Apply to matching slot(s) on tm.mesh.material — but ONLY when the
    // current material is the textured baseMaterial. Vertex-color LODs
    // ignore textures.
    const mat = tm.mesh.material;
    if (mat === tm.baseMaterial) {
      const targets = _findMaterialSlots(mat, desc);
      for (const tex of targets) {
        tex.dispose();
        tex.image = bmp;
        tex.needsUpdate = true;
      }
    }
    tState.currentLod = wantIdx;
  }

  // Detach our root from the scene graph when every tracked mesh is routed
  // through an InstancedMesh slot. three.js then skips this whole subtree
  // during updateMatrixWorld and render-list construction. The instanced-mesh
  // matrix is the only renderer-visible state we still need to push.
  _maybeDetach() {
    if (this._detached || this._disposed) return;
    if (!this.trackedMeshes.length) return;
    for (const tm of this.trackedMeshes) {
      if (!tm._instancedSlot) return; // at least one mesh still needs per-entity draw
    }
    const parent = this.root.parent;
    if (!parent) return;
    this._sceneParent = parent;
    parent.remove(this.root);
    this._detached = true;
  }
  _maybeReattach() {
    if (!this._detached || this._disposed) return;
    // Re-attach as soon as ANY tracked mesh leaves the instanced tier.
    let allInstanced = true;
    for (const tm of this.trackedMeshes) {
      if (!tm._instancedSlot) { allInstanced = false; break; }
    }
    if (allInstanced) return;
    if (this._sceneParent) {
      this._sceneParent.add(this.root);
      // Force a matrixWorld recompute on next frame since the subtree was
      // detached and possibly skipped updates.
      this.root.matrixWorldNeedsUpdate = true;
    }
    this._detached = false;
  }

  // Per-frame update called by the pool. Returns the screen-space pixel size
  // used so callers can use it for further decisions.
  _update(camera, viewportHeight, dt, globalCeilingLod, frustum, animationThrottleDistance) {
    if (this._disposed) return 0;
    let primaryMesh = this.trackedMeshes[0]?.mesh;
    if (!primaryMesh) return 0;
    // Remember scene parent the first time we see it so we can re-attach
    // later even if the root was detached for being fully instanced.
    if (!this._sceneParent && this.root.parent) this._sceneParent = this.root.parent;
    // Lazy matrix update: only invalidated when root.position/rotation/scale
    // changed since last tick. The autoUpdate flag controls this.
    // When detached, this is a no-op walk over an unparented root — still
    // cheap, and keeps matrixWorld current for instance-slot writes.
    if (this.root.matrixWorldNeedsUpdate || this.root.matrixAutoUpdate) {
      this.root.updateMatrixWorld(true);
    }
    const sphere = primaryMesh.geometry?.boundingSphere;
    if (!sphere) return 0;
    const world = _tmpV3.setFromMatrixPosition(primaryMesh.matrixWorld);
    // Cheap distance gate before any frustum work. Entities far enough away
    // get the instanced-tier matrix update only — no LOD decisions, no
    // animation. The "far" threshold maps to a screen-px below the LOD-0
    // visibility floor at the typical viewing FOV.
    const cameraToEntity = world.distanceTo(camera.position);
    const scaleLen = this.root.scale.length();
    const radius = sphere.radius * scaleLen / Math.SQRT2;
    // Skip frustum check + LOD if we're MUCH closer than necessary OR much
    // farther than we can resolve. The frustum test itself is moderately
    // expensive (matrix-vs-sphere per plane).
    // If any tracked mesh is currently in an instanced slot, the GPU
    // vertex shader handles frustum culling for us — skip the CPU sphere
    // test entirely. Per-entity meshes (HERO/MID) still need it.
    const anyInstancedNow = this.trackedMeshes.some((tm) => tm._instancedSlot && tm._instancedSlotIdx >= 0);
    let inFrustum;
    if (anyInstancedNow) {
      inFrustum = true;
    } else {
      _tmpSphere.set(world, radius);
      inFrustum = frustum ? frustum.intersectsSphere(_tmpSphere) : true;
    }
    this._lastInFrustum = inFrustum;
    if (this.root.visible !== inFrustum) this.root.visible = inFrustum;
    if (!inFrustum) {
      // Still push instance matrix in case a slot is active — we want the
      // matrix valid when the camera swings back. Note: we zero it via
      // visible:false on the root, but the InstancedMesh ignores root
      // visibility (it's a sibling in the scene tree). Set the slot matrix
      // to a far-away point so it's outside the camera regardless.
      for (const tm of this.trackedMeshes) {
        if (tm._instancedSlot && tm._instancedSlotIdx >= 0) {
          tm._instancedSlot.setMatrixForSlot(tm._instancedSlotIdx, this.root.matrixWorld);
        }
      }
      this._maybeReattach();
      this._maybeDetach();
      return 0;
    }
    const dist = camera.position.distanceTo(world);
    const fovRad = THREE.MathUtils.degToRad(camera.fov);
    const halfWorld = Math.tan(fovRad / 2) * Math.max(dist, 0.0001);
    const screenPx = (radius / halfWorld) * viewportHeight;
    // Tier routing: HERO/MID/FAR are all decided by which LOD the screen-
    // space picker selects. FAR-distance entities resolve to the unskinned
    // (idx 0) LOD which the pool routes through a shared per-asset
    // InstancedMesh — real 3D geometry, just decimated, so shape survives
    // for non-character meshes (terrain chunks, props, anything).
    const tinyOnScreen = screenPx < 4;
    {
      if (!tinyOnScreen) {
        for (const tm of this.trackedMeshes) {
          const desc = this.asset.meshLodDescs[tm.meshDescIdx];
          if (!desc) continue;
          const targetIdx = _pickMeshLod(desc.lods, screenPx, globalCeilingLod);
          if (targetIdx !== tm.currentLod) this._applyLod(tm, targetIdx);
          for (let ti = 0; ti < tm.texState.length; ti++) {
            const tDesc = this.asset.texLodDescs[ti];
            if (!tDesc) continue;
            const tWant = _pickTexLod(tDesc.lods, screenPx);
            if (tWant !== tm.texState[ti].currentLod) this._applyTexLod(tm, ti, tWant);
          }
        }
      }
    }
    // Push instance matrices for instanced-tier tracked meshes. Also
    // refresh the per-instance world-space bound-sphere center for GPU
    // frustum culling — only when the entity actually moves (root has
    // auto-update on, or is flagged for one-shot rebuild).
    const movable = this.root.matrixAutoUpdate || this._boundDirty;
    for (const tm of this.trackedMeshes) {
      if (tm._instancedSlot && tm._instancedSlotIdx >= 0) {
        tm._instancedSlot.setMatrixForSlot(tm._instancedSlotIdx, this.root.matrixWorld);
        if (movable && tm._instancedBoundRadius != null) {
          const me = this.root.matrixWorld.elements;
          const scale = this.root.scale.length() / Math.SQRT2;
          tm._instancedSlot.setBoundSphereForSlot(
            tm._instancedSlotIdx, me[12], me[13], me[14],
            tm._instancedBoundRadius * scale,
          );
        }
      }
    }
    this._boundDirty = false;
    // If every tracked mesh is now instanced, detach root from the scene
    // graph so three.js stops paying traversal cost on it. Re-attach as soon
    // as any tracked mesh leaves the instanced tier.
    this._maybeReattach();
    this._maybeDetach();
    // Animation throttle: at distance, run mixer at reduced rate. Skip
    // entirely when the entity is instanced (bind pose only).
    const anyInstanced = this.trackedMeshes.some((tm) => !!tm._instancedSlot);
    if (this.animationMixer && !anyInstanced) {
      if (dist > animationThrottleDistance) {
        if ((this._animTickCounter = ((this._animTickCounter || 0) + 1)) % 3 === 0) {
          this.animationMixer.update(dt * 3);
        }
      } else {
        this.animationMixer.update(dt);
      }
    }
    if (this.vrm?.update && dist < animationThrottleDistance && !anyInstanced) this.vrm.update(dt);
    return screenPx;
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    this.root.parent?.remove(this.root);
    // If we were detached for being fully instanced, the root has no parent
    // but _sceneParent still holds the original — nothing to remove there
    // (we already removed ourselves at detach time). Clear the reference.
    this._sceneParent = null;
    this._detached = false;
    // Stop animation.
    if (this.animationAction) this.animationAction.stop();
    this.animationMixer = null;
    this.animationAction = null;
    // Release any instanced-mesh slots we held.
    for (const tm of this.trackedMeshes) {
      if (tm._instancedSlot) tm._instancedSlot.releaseSlot(this);
      if (tm.vcMaterial) tm.vcMaterial.dispose();
    }
    this.trackedMeshes = [];
    this.pool._entities.delete(this);
    this.emit('disposed', this);
  }
}

// Helper: clone a three.js scene with SkinnedMesh skeletons re-bound to the
// cloned bone tree. Crucial: MATERIALS AND GEOMETRIES are shared with the
// source — only the Object3D scene-graph topology and per-mesh skeleton
// objects are unique per entity. Without this every spawned entity allocates
// its own Texture/Material clones, leaking 100s of GPU textures with N=500
// even though they're never used.
function _cloneSkinned(source) {
  const sourceToClone = new Map();
  const cloneRoot = _cloneObject3D(source, sourceToClone);
  // For every SkinnedMesh, build a fresh skeleton with cloned bones.
  cloneRoot.traverse((cm) => {
    if (!cm.isSkinnedMesh) return;
    let sourceSm = null;
    for (const [s, c] of sourceToClone) {
      if (c === cm) { sourceSm = s; break; }
    }
    if (!sourceSm) return;
    const srcSkel = sourceSm.skeleton;
    if (!srcSkel) return;
    const newBones = srcSkel.bones.map((b) => sourceToClone.get(b) || b);
    const newSkel = new THREE.Skeleton(newBones, srcSkel.boneInverses);
    cm.bind(newSkel, cm.bindMatrix);
  });
  return cloneRoot;
}

// Selective clone: preserves shared materials/geometries, copies transforms.
function _cloneObject3D(src, sourceToClone) {
  let copy;
  if (src.isSkinnedMesh) {
    // Share geometry + material; per-entity skeleton is rebuilt above.
    copy = new THREE.SkinnedMesh(src.geometry, src.material);
    copy.bindMode = src.bindMode;
    copy.bindMatrix.copy(src.bindMatrix);
    copy.bindMatrixInverse.copy(src.bindMatrixInverse);
  } else if (src.isMesh) {
    copy = new THREE.Mesh(src.geometry, src.material);
  } else if (src.isBone) {
    copy = new THREE.Bone();
  } else {
    copy = new THREE.Object3D();
  }
  copy.name = src.name;
  copy.position.copy(src.position);
  copy.quaternion.copy(src.quaternion);
  copy.scale.copy(src.scale);
  copy.matrixAutoUpdate = src.matrixAutoUpdate;
  copy.visible = src.visible;
  copy.frustumCulled = src.frustumCulled;
  sourceToClone.set(src, copy);
  for (const child of src.children) {
    copy.add(_cloneObject3D(child, sourceToClone));
  }
  return copy;
}

// LOD picker — same logic as the inline demo, plus a ceiling clamp.
function _pickMeshLod(lods, screenPx, ceilingIdx) {
  const thresholds = [80, 200, 400, 800, 1400];
  let i = 0;
  for (const t of thresholds) { if (screenPx > t) i++; else break; }
  if (ceilingIdx != null) i = Math.min(i, ceilingIdx);
  return Math.min(i, lods.length - 1);
}
function _pickTexLod(lods, screenPx) {
  const target = Math.max(64, screenPx);
  let bestIdx = 0;
  for (let i = 0; i < lods.length; i++) {
    if (lods[i].width <= target * 2) bestIdx = i;
  }
  return bestIdx;
}

// Same texture-slot resolver as the inline demo.
function _findMaterialSlots(mat, texEntry) {
  if (!mat) return [];
  const slots = ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'emissiveMap'];
  const out = new Set();
  for (const slot of slots) {
    const t = mat[slot];
    if (!t) continue;
    const tname = t.name || '';
    if (tname && texEntry.name && tname === texEntry.name) out.add(t);
  }
  if (out.size) return [...out];
  const nm = (texEntry.name || '').toLowerCase();
  if (nm.includes('normal') && mat.normalMap) out.add(mat.normalMap);
  if ((nm.includes('metallic') || nm.includes('roughness'))) {
    if (mat.roughnessMap) out.add(mat.roughnessMap);
    if (mat.metalnessMap) out.add(mat.metalnessMap);
  }
  if (!out.size && mat.map) out.add(mat.map);
  return [...out];
}

// --- ModelPool: the public facade -----------------------------------------
export class ModelPool extends Emitter {
  constructor(opts = {}) {
    super();
    this.scene = opts.scene;
    this.renderer = opts.renderer;
    this.camera = opts.camera;
    this.targetFps = opts.targetFps ?? 50;
    this.byteBudget = opts.byteBudget ?? 256 * 1024 * 1024;
    this.maxConcurrentFetches = opts.maxConcurrentFetches ?? 6;
    this.animationThrottleDistance = opts.animationThrottleDistance ?? 25;
    this._assets = new Map(); // url -> Asset
    this._entities = new Set();
    this._nextEntityId = 0;
    this._totalBytes = 0;
    this._byteLog = new Map(); // assetUrl -> { url -> bytes }
    this._loadQueue = new Map(); // dedupe key -> Promise
    this._inFlight = 0;
    this._pending = []; // queued tasks { key, run, resolve, reject }
    this._fpsEma = 60;
    this._lastTick = performance.now();
    this._currentCeilingLod = null; // null = unrestricted
    this._frustum = new THREE.Frustum();
    this._tmpMatrix = new THREE.Matrix4();
    // Stats snapshot — refreshed each tick, exposed via getStats().
    this._stats = { fps: 0, entities: 0, drawCalls: 0, ceilingLod: null, bytes: 0, assets: 0, inFlight: 0, hero: 0, mid: 0, far: 0 };
    // Shared InstancedMesh slots: key `${assetUrl}|${meshDescIdx}|${lodIdx}` -> InstancedSlot.
    this._instancedSlots = new Map();
    // Tier thresholds (in screen pixels of the entity's bounding sphere).
    // The three tiers are entirely a function of which LOD the picker
    // selects, NOT a separate routing layer:
    //   HERO: top-of-ladder textured LODs (per-entity SkinnedMesh draws).
    //   MID:  middle textured / vertcolor LODs (still per-entity draws, but
    //         the geometry is decimated and the material is cheaper).
    //   FAR:  unskinned LOD (routes through per-asset InstancedMesh; real 3D
    //         geometry preserved — works for terrain chunks, props, anything).
    // We track these in the HUD only for visibility; routing happens via the
    // standard LOD picker.
    this.heroPx = opts.heroPx ?? 200;
    this.midPx = opts.midPx ?? 30;
    this.heroCap = opts.heroCap ?? 20;
    // Worker pool for sibling-LOD fetch + decode. Defaults to 4 workers
    // (more = more concurrent decodes; each holds one three.js instance so
    // memory grows linearly). Set to 0 to disable and fall back to
    // main-thread decode.
    this._workerCount = opts.workerCount ?? 4;
    this._workers = [];
    this._workerRR = 0; // round-robin cursor
    this._workerPending = new Map(); // id -> {resolve, reject}
    this._workerNextId = 0;
    if (this._workerCount > 0 && typeof Worker !== 'undefined') {
      try {
        for (let i = 0; i < this._workerCount; i++) {
          const workerUrl = new URL('./lod-worker.js', import.meta.url);
          const w = new Worker(workerUrl, { type: 'module' });
          w.addEventListener('message', (ev) => {
            const m = ev.data;
            if (m && m.id === 0 && m.ready) {
              if (!m.ok) console.error('[pool] worker init failed:', m.error);
              else console.log('[pool] worker ready');
              return;
            }
            this._onWorkerMessage(m);
          });
          w.addEventListener('error', (e) => {
            const detail = {
              message: e.message || '(no message)',
              filename: e.filename || '(no filename)',
              lineno: e.lineno,
              colno: e.colno,
              error: e.error ? (e.error.stack || String(e.error)) : '(no error obj)',
              workerUrl: String(workerUrl),
            };
            console.error('[pool] worker error', JSON.stringify(detail));
          });
          w.addEventListener('messageerror', (e) => console.error('[pool] worker messageerror', String(e)));
          this._workers.push(w);
        }
      } catch (e) {
        console.warn('[pool] worker init failed, falling back to main-thread decode', e);
        this._workers = [];
      }
    }
  }

  _onWorkerMessage(msg) {
    const pend = this._workerPending.get(msg.id);
    if (!pend) return;
    this._workerPending.delete(msg.id);
    if (msg.ok) pend.resolve(msg.payload);
    else pend.reject(new Error(msg.error || 'worker decode failed'));
  }

  // Fetch + decode a sibling LOD GLB in a worker; returns a Promise<payload>
  // where payload is { attrs, index, boundingSphere, boundingBox, bytes }.
  _workerFetchLod(url, decodeAABB) {
    if (!this._workers.length) return null;
    const id = ++this._workerNextId;
    const w = this._workers[this._workerRR];
    this._workerRR = (this._workerRR + 1) % this._workers.length;
    return new Promise((resolve, reject) => {
      this._workerPending.set(id, { resolve, reject });
      w.postMessage({ id, url, decodeAABB });
    });
  }

  // Streaming variant: same worker, different message shape. Worker fetches
  // a byte-range slice of `url`, composes a sub-GLB using `jsonSpec`, parses
  // it, and returns the same payload shape as `_workerFetchLod`.
  _workerFetchStreamingLod(url, byteOffset, byteLength, jsonSpec, decodeAABB, binBase) {
    if (!this._workers.length) return null;
    const id = ++this._workerNextId;
    const w = this._workers[this._workerRR];
    this._workerRR = (this._workerRR + 1) % this._workers.length;
    return new Promise((resolve, reject) => {
      this._workerPending.set(id, { resolve, reject });
      w.postMessage({ id, url, streaming: true, byteOffset, byteLength, jsonSpec, decodeAABB, binBase });
    });
  }

  // Rebuild a BufferGeometry on the main thread from a worker payload.
  // No heavy work here — typed arrays were transferred, so this is just
  // attribute wiring.
  static _buildGeometryFromPayload(payload) {
    const geo = new THREE.BufferGeometry();
    for (const k of Object.keys(payload.attrs)) {
      const a = payload.attrs[k];
      geo.setAttribute(k, new THREE.BufferAttribute(a.array, a.itemSize, !!a.normalized));
    }
    if (payload.index) {
      geo.setIndex(new THREE.BufferAttribute(payload.index, 1));
    }
    if (payload.boundingSphere) {
      geo.boundingSphere = new THREE.Sphere(
        new THREE.Vector3().fromArray(payload.boundingSphere.center),
        payload.boundingSphere.radius
      );
    }
    if (payload.boundingBox) {
      geo.boundingBox = new THREE.Box3(
        new THREE.Vector3().fromArray(payload.boundingBox.min),
        new THREE.Vector3().fromArray(payload.boundingBox.max)
      );
    }
    return geo;
  }

  // Get-or-create an InstancedSlot for an (asset, meshDescIdx, lod) tuple.
  // Returns null if the LOD isn't suitable for instancing (currently only
  // 'unskinned' LODs qualify — they have no per-instance bone state).
  _getInstancedSlot(asset, meshDescIdx, lodIdx) {
    const desc = asset.meshLodDescs[meshDescIdx];
    if (!desc) return null;
    const lod = desc.lods[lodIdx];
    if (!lod || (lod.kind || 'textured') !== 'unskinned') return null;
    const key = `${asset.url}|${meshDescIdx}|${lodIdx}`;
    let slot = this._instancedSlots.get(key);
    if (slot) return slot;
    const geo = asset.geoCache.get(`${desc.meshIndex}:${desc.primIndex}:${lodIdx}`);
    if (!geo) return null; // not loaded yet
    // Build a shared vertex-color material for the InstancedMesh.
    const mat = new THREE.MeshLambertMaterial({ vertexColors: true });
    mat.onBeforeCompile = (shader) => {
      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <color_fragment>',
        `#if defined( USE_COLOR_ALPHA )
          diffuseColor.rgb *= pow(vColor.rgb, vec3(2.2));
          diffuseColor.a *= vColor.a;
        #elif defined( USE_COLOR )
          diffuseColor.rgb *= pow(vColor, vec3(2.2));
        #endif`
      );
    };
    slot = new InstancedSlot(this, asset, meshDescIdx, lodIdx, geo, mat);
    this._instancedSlots.set(key, slot);
    // Attach the instanced mesh to the same scene as entities.
    this.scene.add(slot.mesh);
    return slot;
  }

  // Get-or-load an asset; idempotent.
  async _resolveAsset(url) {
    let a = this._assets.get(url);
    if (!a) {
      a = new Asset(this, url);
      this._assets.set(url, a);
      a.ready.then(() => this.emit('asset-ready', a)).catch((e) => this.emit('asset-error', { asset: a, error: e }));
    }
    return a;
  }

  // Spawn an entity from a URL. Returns an Entity handle synchronously; the
  // Entity emits 'ready' once loading completes.
  spawn(url, opts = {}) {
    if (!url) throw new Error('spawn(): url required');
    const assetPromise = this._resolveAsset(url);
    // Build a placeholder entity tied to a yet-to-resolve asset.
    // We attach to the entity once the asset is ready inside Entity._bootstrap.
    const placeholder = { _disposed: false };
    let actualEntity = null;
    // Wrap into a lightweight proxy so caller can listen on events even
    // before bootstrap finishes.
    const proxy = new Emitter();
    proxy.root = new THREE.Object3D();
    proxy.root.name = `pending_${++this._nextEntityId}`;
    proxy.dispose = () => {
      placeholder._disposed = true;
      if (actualEntity) actualEntity.dispose();
      else proxy.root.parent?.remove(proxy.root);
    };
    assetPromise.then((asset) => {
      if (placeholder._disposed) return;
      actualEntity = new Entity(this, asset, opts);
      this._entities.add(actualEntity);
      // Stitch: replace proxy.root with actualEntity.root in the parent.
      const parent = proxy.root.parent;
      if (parent) {
        parent.add(actualEntity.root);
        parent.remove(proxy.root);
      }
      // Re-forward events.
      actualEntity.on('ready', (e) => proxy.emit('ready', e));
      actualEntity.on('lod-changed', (e) => proxy.emit('lod-changed', e));
      actualEntity.on('disposed', (e) => proxy.emit('disposed', e));
      actualEntity.on('error', (e) => proxy.emit('error', e));
      // Expose useful entity props on the proxy.
      proxy.actualEntity = actualEntity;
      proxy.root = actualEntity.root;
    }).catch((e) => proxy.emit('error', e));
    return proxy;
  }

  // Per-frame update: call from your render loop AFTER advancing camera.
  update() {
    const tUpdate0 = performance.now();
    const now = tUpdate0;
    const dt = (now - this._lastTick) / 1000;
    this._lastTick = now;
    // EMA FPS over ~1s.
    const instFps = dt > 0 ? 1 / dt : 60;
    this._fpsEma = this._fpsEma * 0.95 + instFps * 0.05;
    // Adaptive ceiling: every 30 frames roughly, if FPS far below target
    // lower ceiling by 1; if comfortably above, raise it by 1.
    if (!this._fpsAdjustCountdown) this._fpsAdjustCountdown = 30;
    this._fpsAdjustCountdown--;
    if (this._fpsAdjustCountdown <= 0) {
      this._fpsAdjustCountdown = 30;
      const target = this.targetFps;
      if (this._fpsEma < target - 5) {
        // Multi-knob pressure response:
        //  1. Drop LOD ceiling first.
        //  2. If still under target, shrink midPx (push more entities to FAR
        //     impostor tier).
        //  3. If still under target, reduce heroCap (smaller hero count).
        let changed = false;
        const nextCeil = (this._currentCeilingLod ?? 5) - 1;
        if (nextCeil >= 0 && this._currentCeilingLod !== nextCeil) {
          this._currentCeilingLod = nextCeil;
          changed = true;
        } else if (this.midPx < 200) {
          // Expand FAR range — more entities collapse to unskinned tier.
          this.midPx = Math.min(200, this.midPx + 20);
          changed = true;
        } else if (this.heroCap > 5) {
          this.heroCap = Math.max(5, this.heroCap - 5);
          changed = true;
        }
        if (changed) this.emit('budget-adjust', {
          ceiling: this._currentCeilingLod, midPx: this.midPx, heroCap: this.heroCap, fps: this._fpsEma,
        });
      } else if (this._fpsEma > target + 5) {
        // Headroom — relax knobs in reverse priority.
        let changed = false;
        if (this.heroCap < 20) {
          this.heroCap = Math.min(20, this.heroCap + 5);
          changed = true;
        } else if (this.midPx > 30) {
          this.midPx = Math.max(30, this.midPx - 10);
          changed = true;
        } else if (this._currentCeilingLod != null) {
          const next = this._currentCeilingLod + 1;
          if (next >= 5) this._currentCeilingLod = null;
          else this._currentCeilingLod = next;
          changed = true;
        }
        if (changed) this.emit('budget-adjust', {
          ceiling: this._currentCeilingLod, midPx: this.midPx, heroCap: this.heroCap, fps: this._fpsEma,
        });
      }
    }
    // Build frustum once per frame.
    const tFrustum0 = performance.now();
    this.camera.updateMatrixWorld();
    this._tmpMatrix.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
    this._frustum.setFromProjectionMatrix(this._tmpMatrix);
    // Publish projView to every InstancedSlot's shader uniform so the GPU
    // can run the per-instance frustum cull pass.
    for (const slot of this._instancedSlots.values()) {
      slot._uniforms.projViewMatrix.value.copy(this._tmpMatrix);
    }
    const vh = this.renderer.domElement.clientHeight;
    const tFrustum1 = performance.now();
    // Cull + update each entity. Tally per-tier counts for the HUD.
    let visible = 0, hero = 0, mid = 0, far = 0;
    for (const e of this._entities) {
      const px = e._update(this.camera, vh, dt, this._currentCeilingLod, this._frustum, this.animationThrottleDistance);
      if (px > 0) visible++;
      if (px > this.heroPx) hero++;
      else if (px > this.midPx) mid++;
      else if (px > 0) far++;
    }
    const tEntities1 = performance.now();
    this._stats.hero = hero;
    this._stats.mid = mid;
    this._stats.far = far;
    this._stats.msFrustum = tFrustum1 - tFrustum0;
    this._stats.msEntities = tEntities1 - tFrustum1;
    // Maintain byte budget. If still over budget after eviction (because
    // active LODs can't be evicted), tighten midPx so more entities drop
    // to the unskinned tier — that releases higher-LOD references and
    // unlocks eviction next sweep. Geometry shape is still preserved
    // because the unskinned tier is a real (decimated) mesh, not a sprite.
    this._enforceBudget();
    if (this._totalBytes > this.byteBudget && this.midPx < 200) {
      // Larger midPx → wider FAR catch (entities up to midPx screen-size).
      this.midPx = Math.min(200, this.midPx + 5);
      this.emit('budget-adjust', { reason: 'over-budget', midPx: this.midPx, bytes: this._totalBytes, budget: this.byteBudget });
    }
    // Stats.
    this._stats.fps = this._fpsEma;
    this._stats.entities = this._entities.size;
    this._stats.visible = visible;
    this._stats.drawCalls = this.renderer.info?.render?.calls ?? 0;
    this._stats.ceilingLod = this._currentCeilingLod;
    this._stats.bytes = this._totalBytes;
    this._stats.assets = this._assets.size;
    this._stats.inFlight = this._inFlight;
    this._stats.msTotal = performance.now() - tUpdate0;
    this.emit('fps', this._stats);
  }

  // Public stats accessor (cheap, no allocation).
  getStats() { return this._stats; }

  // --- Byte tracking + budget ---------------------------------------------
  _trackBytes(assetUrl, url, bytes) {
    this._totalBytes += bytes;
    let log = this._byteLog.get(assetUrl);
    if (!log) { log = new Map(); this._byteLog.set(assetUrl, log); }
    log.set(url, bytes);
  }
  _untrackBytes(assetUrl, url) {
    const log = this._byteLog.get(assetUrl);
    if (!log) return;
    const b = log.get(url) || 0;
    this._totalBytes -= b;
    log.delete(url);
  }
  _enforceBudget() {
    if (this._totalBytes <= this.byteBudget) return;
    // LRU eviction with hard byte budget.
    //
    // 1. Build inUse set of (asset|kind:descIdx:lodIdx) keys currently bound
    //    to a live Entity's tracked mesh / texture slots.
    // 2. Gather all evict-eligible cached entries (non-inline, not in inUse)
    //    with their lastTouched timestamp and byte cost.
    // 3. Sort ascending by lastTouched (oldest = first to go).
    // 4. Evict one-by-one until under budget OR no candidates left,
    //    emitting `budget-pressure` per eviction so callers can observe
    //    working-set churn (not just one summary).
    const inUse = new Set();
    for (const e of this._entities) {
      for (const tm of e.trackedMeshes) {
        const d = e.asset.meshLodDescs[tm.meshDescIdx];
        if (d) inUse.add(`${e.asset.url}|geo|${d.meshIndex}:${d.primIndex}:${tm.currentLod}`);
        for (let ti = 0; ti < tm.texState.length; ti++) {
          const td = e.asset.texLodDescs[ti];
          if (td) inUse.add(`${e.asset.url}|tex|${td.textureIndex}:${tm.texState[ti].currentLod}`);
        }
      }
    }

    // Gather candidates. We walk descriptors (which know the inline flag and
    // byte weight) rather than the raw cache so we never accidentally
    // consider an inline / root-baked entry.
    const now = _now();
    const candidates = [];
    for (const asset of this._assets.values()) {
      for (let di = 0; di < asset.meshLodDescs.length; di++) {
        const desc = asset.meshLodDescs[di];
        for (let li = 0; li < desc.lods.length; li++) {
          if (desc.lods[li].inline) continue;
          const cacheKey = `${desc.meshIndex}:${desc.primIndex}:${li}`;
          if (!asset.geoCache.has(cacheKey)) continue; // not cached
          const inUseKey = `${asset.url}|geo|${cacheKey}`;
          if (inUse.has(inUseKey)) continue;
          const lastTouched = asset.geoTouch.get(cacheKey) || 0;
          const bytes = asset.byteWeights.get(cacheKey) || desc.lods[li].bytes || 0;
          candidates.push({ asset, kind: 'geo', descIdx: di, lodIdx: li, cacheKey, lastTouched, bytes });
        }
      }
      for (let di = 0; di < asset.texLodDescs.length; di++) {
        const desc = asset.texLodDescs[di];
        for (let li = 0; li < desc.lods.length; li++) {
          if (desc.lods[li].inline) continue;
          const cacheKey = `${desc.textureIndex}:${li}`;
          if (!asset.texCache.has(cacheKey)) continue;
          const inUseKey = `${asset.url}|tex|${cacheKey}`;
          if (inUse.has(inUseKey)) continue;
          const lastTouched = asset.texTouch.get(cacheKey) || 0;
          const bytes = asset.byteWeights.get(`tex:${cacheKey}`) || desc.lods[li].bytes || 0;
          candidates.push({ asset, kind: 'tex', descIdx: di, lodIdx: li, cacheKey, lastTouched, bytes });
        }
      }
    }
    // Oldest first.
    candidates.sort((a, b) => a.lastTouched - b.lastTouched);
    for (const c of candidates) {
      if (this._totalBytes <= this.byteBudget) break;
      const ok = c.kind === 'geo'
        ? c.asset.evictMeshLod(c.descIdx, c.lodIdx)
        : c.asset.evictTexLod(c.descIdx, c.lodIdx);
      if (!ok) continue;
      // _trackBytes uses (assetUrl, url, bytes); we never recorded a precise
      // url here (per-LOD fetch already called _trackBytes/_untrackBytes via
      // the fetch wrappers in some paths but not all). The byteWeights map
      // is the authoritative size for the cached resource — subtract that.
      this._totalBytes -= c.bytes;
      if (this._totalBytes < 0) this._totalBytes = 0;
      this.emit('budget-pressure', {
        key: `${c.asset.url}|${c.kind}|${c.cacheKey}`,
        kind: c.kind,
        lastTouched: c.lastTouched,
        ageMs: c.lastTouched ? (now - c.lastTouched) : null,
        bytes: c.bytes,
        remaining: this._totalBytes,
      });
    }
  }

  // --- Bounded concurrent fetch queue --------------------------------------
  _enqueue(key, run) {
    const existing = this._loadQueue.get(key);
    if (existing) return existing;
    const p = new Promise((resolve, reject) => {
      const task = { key, run, resolve, reject };
      if (this._inFlight < this.maxConcurrentFetches) this._runTask(task);
      else this._pending.push(task);
    });
    this._loadQueue.set(key, p);
    p.finally(() => this._loadQueue.delete(key));
    return p;
  }
  async _runTask(task) {
    this._inFlight++;
    try {
      const r = await task.run();
      task.resolve(r);
    } catch (e) {
      task.reject(e);
    } finally {
      this._inFlight--;
      if (this._pending.length && this._inFlight < this.maxConcurrentFetches) {
        this._runTask(this._pending.shift());
      }
    }
  }
}
