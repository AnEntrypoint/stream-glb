// Web Worker: fetches a sibling LOD GLB, parses it with GLTFLoader +
// MeshoptDecoder, decodes any meshopt quantization, runs the same
// _bakeQuantizeDecode logic the main-thread path used, and posts back
// transferable typed arrays. Main thread rebuilds the BufferGeometry from
// the payload — that step is O(slot allocations), no heavy work.
//
// The worker is a MODULE worker (`type: 'module'`) so we can `import`
// three.js + GLTFLoader from the same CDN versions the page uses.
//
// NOTE: static top-level `import` from cross-origin CDN URLs in a module
// worker silently fails in some Chromium versions (the error event arrives
// with `message: ''` and no `error` object — completely undiagnosable from
// the parent page). We side-step that by doing DYNAMIC `import()` inside a
// try/catch so we can post the real error back to the main thread before
// the worker dies.

let THREE = null;
let GLTFLoader = null;
let MeshoptDecoder = null;
let loader = null;
let readyResolve;
const readyPromise = new Promise((r) => { readyResolve = r; });

(async () => {
  try {
    // Use esm.sh which rewrites the bare specifier `three` (used inside
    // GLTFLoader / meshopt_decoder) into a real URL — module workers do
    // NOT inherit the page's <script type="importmap">, so vanilla
    // cdn.jsdelivr.net URLs fail with "Failed to resolve module specifier 'three'".
    // The ?deps pin keeps every import on the same three.js version so we
    // don't end up with two THREE.* runtimes in the worker.
    const threeMod = await import('https://esm.sh/three@0.170.0');
    THREE = threeMod;
    const gltfMod = await import('https://esm.sh/three@0.170.0/examples/jsm/loaders/GLTFLoader.js?deps=three@0.170.0');
    GLTFLoader = gltfMod.GLTFLoader;
    const meshoptMod = await import('https://esm.sh/three@0.170.0/examples/jsm/libs/meshopt_decoder.module.js?deps=three@0.170.0');
    MeshoptDecoder = meshoptMod.MeshoptDecoder;
    loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    readyResolve(true);
    self.postMessage({ id: 0, ok: true, ready: true });
  } catch (e) {
    self.postMessage({ id: 0, ok: false, ready: true, error: 'worker init: ' + String(e && (e.stack || e.message || e)) });
    readyResolve(false);
  }
})();

self.addEventListener('error', (e) => {
  try {
    self.postMessage({ id: 0, ok: false, ready: true, error: 'worker self.error: ' + (e.message || '') + ' @ ' + (e.filename || '') + ':' + (e.lineno || '') });
  } catch {}
});

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

// Extract attributes from a geometry into a serializable payload with
// transferable typed-array buffers.
function extractGeometry(geo) {
  const attrs = {};
  for (const k of Object.keys(geo.attributes)) {
    const a = geo.attributes[k];
    // Force into a flat Float32Array — _bakeQuantizeDecode already did this
    // for position/normal/tangent. For color/uv/skinWeight/skinIndex we may
    // still have other types — copy them out flat too so the main thread
    // doesn't need attribute-type knowledge.
    let arr;
    if (a.isInterleavedBufferAttribute || !(a.array instanceof Float32Array)) {
      arr = new Float32Array(a.count * a.itemSize);
      for (let i = 0; i < a.count; i++) {
        if (a.itemSize >= 1) arr[i * a.itemSize + 0] = a.getX(i);
        if (a.itemSize >= 2) arr[i * a.itemSize + 1] = a.getY(i);
        if (a.itemSize >= 3) arr[i * a.itemSize + 2] = a.getZ(i);
        if (a.itemSize >= 4) arr[i * a.itemSize + 3] = a.getW(i);
      }
    } else {
      arr = new Float32Array(a.array.buffer.slice(a.array.byteOffset, a.array.byteOffset + a.array.byteLength));
    }
    attrs[k] = { array: arr, itemSize: a.itemSize, normalized: a.normalized };
  }
  let index = null;
  if (geo.index) {
    const ia = geo.index.array;
    // Copy to a fresh buffer so we can transfer it without worrying about
    // shared underlying ArrayBuffers (meshopt sometimes interleaves).
    if (ia instanceof Uint32Array) index = new Uint32Array(ia);
    else if (ia instanceof Uint16Array) index = new Uint16Array(ia);
    else index = new Uint32Array(ia);
  }
  const bs = geo.boundingSphere;
  const bb = geo.boundingBox;
  return {
    attrs,
    index,
    boundingSphere: bs ? { center: [bs.center.x, bs.center.y, bs.center.z], radius: bs.radius } : null,
    boundingBox: bb ? { min: [bb.min.x, bb.min.y, bb.min.z], max: [bb.max.x, bb.max.y, bb.max.z] } : null,
  };
}

function payloadTransferables(payload) {
  const list = [];
  for (const k of Object.keys(payload.attrs)) list.push(payload.attrs[k].array.buffer);
  if (payload.index) list.push(payload.index.buffer);
  return list;
}

// Build a valid GLB byte buffer from a glTF JSON spec + a BIN payload. Used
// by the streaming-GLB code path: the worker fetches a byte-range slice of
// the source GLB and wraps it in a tiny standalone GLB referring to just
// one LOD's accessors.
function composeSubGLB(jsonSpec, binBytes) {
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
  const total = 12 + 8 + jsonBytes.byteLength + 8 + binLen;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, 0x46546c67, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  let off = 12;
  dv.setUint32(off, jsonBytes.byteLength, true); off += 4;
  dv.setUint32(off, 0x4e4f534a, true); off += 4;
  out.set(jsonBytes, off); off += jsonBytes.byteLength;
  dv.setUint32(off, binLen, true); off += 4;
  dv.setUint32(off, 0x004e4942, true); off += 4;
  out.set(binBytes, off);
  return out;
}

self.addEventListener('message', async (ev) => {
  const { id, url, decodeAABB, streaming, byteOffset, byteLength, jsonSpec, binBase } = ev.data;
  try {
    const ok = await readyPromise;
    if (!ok || !loader) throw new Error('worker not initialized');
    let buf;
    if (streaming) {
      // Range-request the LOD's contiguous byte slice from the source GLB.
      // byteOffset is RELATIVE to the BIN chunk; binBase is the file-offset
      // where the BIN payload starts (after GLB header + JSON chunk).
      const absStart = (binBase || 0) + byteOffset;
      const absEnd = absStart + byteLength - 1;
      const res = await fetch(url, { headers: { Range: `bytes=${absStart}-${absEnd}` } });
      if (!res.ok && res.status !== 206) throw new Error(`fetch ${url} range ${absStart}-${absEnd}: ${res.status}`);
      let bin = new Uint8Array(await res.arrayBuffer());
      if (res.status === 200 && bin.byteLength > byteLength) {
        bin = bin.subarray(absStart, absStart + byteLength);
      }
      buf = composeSubGLB(jsonSpec, bin).buffer;
    } else {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
      buf = await res.arrayBuffer();
    }
    const gltf = await new Promise((resolve, reject) => {
      loader.parse(buf, '', resolve, reject);
    });
    let srcMesh = null;
    gltf.scene.updateMatrixWorld(true);
    gltf.scene.traverse((c) => { if (c.isMesh && !srcMesh) srcMesh = c; });
    if (!srcMesh) throw new Error('no mesh in LOD sibling');
    _bakeQuantizeDecode(srcMesh.geometry, srcMesh.matrixWorld, decodeAABB);
    const payload = extractGeometry(srcMesh.geometry);
    payload.bytes = buf.byteLength;
    self.postMessage({ id, ok: true, payload }, payloadTransferables(payload));
  } catch (e) {
    self.postMessage({ id, ok: false, error: String(e && e.message || e) });
  }
});
