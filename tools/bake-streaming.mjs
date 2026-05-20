#!/usr/bin/env node
// Phase 2 streaming baker — single-file output.
//
// Produces ONE model.streaming.glb per input GLB/VRM. Every LOD (mesh + texture)
// is packed as bufferViews inside the single BIN chunk. The default scene/mesh
// primitive attributes point at the LOWEST mesh LOD; the default image points
// at the smallest webp LOD. Higher LODs live in the file as opaque bufferView
// regions; their locations are described by extras.LOCAL_progressive so the
// runtime can range-fetch a single contiguous slice per LOD upgrade.
//
// Standard glTF loaders (no extension awareness) can open this file and render
// the lowest LOD.

import { NodeIO, BufferUtils } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { simplify, cloneDocument, prune, dedup, meshopt } from '@gltf-transform/functions';
import { MeshoptSimplifier, MeshoptEncoder, MeshoptDecoder } from 'meshoptimizer';
import draco3dgltf from 'draco3dgltf';
import sharp from 'sharp';
import { mkdir, writeFile, readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');

const INPUT = process.argv[2] || path.join(repoRoot, 'model.glb');
const inputBase = path.basename(INPUT, path.extname(INPUT));
const DEFAULT_OUT = path.join(
  repoRoot,
  'examples/local-progressive',
  inputBase === 'model' ? 'output' : `output_${inputBase}`,
);
const OUT_DIR = process.argv[3] || DEFAULT_OUT;

// LOD plan — mirror bake-progressive.mjs.
const MESH_LOD_RATIOS = [1.0, 0.4, 0.15];
const EXTRA_LOD_STAGES = [
  { ratio: 0.04, kind: 'vertcolor' },
  { ratio: 0.01, kind: 'unskinned' },
];
// Texture LOD pyramid sizes. MAX_TEX_SIZE caps the largest face so individual
// bakes can be kept under GitHub's 50MB warning threshold for problematic
// inputs. Set via env: MAX_TEX_SIZE=1024 node tools/bake-streaming.mjs ...
const MAX_TEX_SIZE = parseInt(process.env.MAX_TEX_SIZE || '2048', 10);
const TEX_LOD_SIZES = [2048, 1024, 512, 256].filter((s) => s <= MAX_TEX_SIZE);

// ---------- sRGB conversion + vertex-color baking (copied from bake-progressive) ----------
const SRGB_TO_LINEAR = new Float32Array(256);
const LINEAR_TO_SRGB_LUT = new Uint8Array(4096);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  SRGB_TO_LINEAR[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
for (let i = 0; i < 4096; i++) {
  const lin = i / 4095;
  const enc = lin <= 0.0031308 ? lin * 12.92 : 1.055 * Math.pow(lin, 1 / 2.4) - 0.055;
  LINEAR_TO_SRGB_LUT[i] = Math.min(255, Math.max(0, Math.round(enc * 255)));
}
function linearToSrgbByte(lin) {
  if (lin <= 0) return 0;
  if (lin >= 1) return 255;
  return LINEAR_TO_SRGB_LUT[Math.min(4095, Math.max(0, Math.round(lin * 4095)))];
}
function sampleLinear(rgbaPixels, width, height, u, v) {
  let uu = u - Math.floor(u);
  let vv = v - Math.floor(v);
  const x = Math.min(width - 1, Math.max(0, Math.floor(uu * width)));
  const y = Math.min(height - 1, Math.max(0, Math.floor(vv * height)));
  const i = (y * width + x) * 4;
  return [
    SRGB_TO_LINEAR[rgbaPixels[i]],
    SRGB_TO_LINEAR[rgbaPixels[i + 1]],
    SRGB_TO_LINEAR[rgbaPixels[i + 2]],
  ];
}
function triangleAvgColorLinear(uvArr, posArr, ia, ib, ic, baseRGBA) {
  const u0 = uvArr[ia * 2], v0 = uvArr[ia * 2 + 1];
  const u1 = uvArr[ib * 2], v1 = uvArr[ib * 2 + 1];
  const u2 = uvArr[ic * 2], v2 = uvArr[ic * 2 + 1];
  let area3d = 1;
  if (posArr) {
    const ax = posArr[ia * 3], ay = posArr[ia * 3 + 1], az = posArr[ia * 3 + 2];
    const bx = posArr[ib * 3], by = posArr[ib * 3 + 1], bz = posArr[ib * 3 + 2];
    const cx = posArr[ic * 3], cy = posArr[ic * 3 + 1], cz = posArr[ic * 3 + 2];
    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    area3d = Math.max(1e-9, 0.5 * Math.sqrt(nx * nx + ny * ny + nz * nz));
  }
  const uvArea = Math.abs((u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)) * 0.5;
  const texelArea = uvArea * baseRGBA.width * baseRGBA.height;
  const N = Math.max(4, Math.min(32, Math.ceil(Math.sqrt(texelArea / 4))));
  let sr = 0, sg = 0, sb = 0, samples = 0;
  for (let i = 0; i <= N; i++) {
    for (let j = 0; j <= N - i; j++) {
      const w0 = i / N;
      const w1 = j / N;
      const w2 = 1 - w0 - w1;
      const u = u0 * w0 + u1 * w1 + u2 * w2;
      const v = v0 * w0 + v1 * w1 + v2 * w2;
      const lin = sampleLinear(baseRGBA.data, baseRGBA.width, baseRGBA.height, u, v);
      sr += lin[0]; sg += lin[1]; sb += lin[2]; samples++;
    }
  }
  return { r: sr / samples, g: sg / samples, b: sb / samples, area: area3d };
}
function buildAveragedVertexColors(simpPrim, baseRGBA, vertCount) {
  const colorArr = new Uint8Array(vertCount * 4);
  const idxAcc = simpPrim.getIndices();
  const uvAcc = simpPrim.getAttribute('TEXCOORD_0');
  const posAcc = simpPrim.getAttribute('POSITION');
  if (!baseRGBA || !uvAcc) {
    for (let v = 0; v < vertCount; v++) colorArr.set([180, 180, 180, 255], v * 4);
    return colorArr;
  }
  const uv = uvAcc.getArray();
  const pos = posAcc?.getArray();
  const acc = new Float64Array(vertCount * 4);
  const idx = idxAcc?.getArray();
  const triangleIter = (a, b, c) => {
    const col = triangleAvgColorLinear(uv, pos, a, b, c, baseRGBA);
    const w = col.area;
    for (const vi of [a, b, c]) {
      acc[vi * 4] += col.r * w;
      acc[vi * 4 + 1] += col.g * w;
      acc[vi * 4 + 2] += col.b * w;
      acc[vi * 4 + 3] += w;
    }
  };
  if (!idx) for (let v = 0; v < vertCount; v += 3) triangleIter(v, v + 1, v + 2);
  else for (let t = 0; t < idx.length; t += 3) triangleIter(idx[t], idx[t + 1], idx[t + 2]);
  for (let v = 0; v < vertCount; v++) {
    const w = acc[v * 4 + 3];
    if (w > 1e-9) {
      colorArr[v * 4] = linearToSrgbByte(acc[v * 4] / w);
      colorArr[v * 4 + 1] = linearToSrgbByte(acc[v * 4 + 1] / w);
      colorArr[v * 4 + 2] = linearToSrgbByte(acc[v * 4 + 2] / w);
    } else {
      const lin = sampleLinear(baseRGBA.data, baseRGBA.width, baseRGBA.height, uv[v * 2], uv[v * 2 + 1]);
      colorArr[v * 4] = linearToSrgbByte(lin[0]);
      colorArr[v * 4 + 1] = linearToSrgbByte(lin[1]);
      colorArr[v * 4 + 2] = linearToSrgbByte(lin[2]);
    }
    colorArr[v * 4 + 3] = 255;
  }
  return colorArr;
}

// ---------- GLB helpers ----------
function writeGlbBlob(json, binBytes) {
  const jsonStr = JSON.stringify(json);
  const jsonBuf = new TextEncoder().encode(jsonStr);
  const jsonPad = (4 - (jsonBuf.length % 4)) % 4;
  const binPad = (4 - (binBytes.length % 4)) % 4;
  const jsonChunkLen = jsonBuf.length + jsonPad;
  const binChunkLen = binBytes.length + binPad;
  const total = 12 + 8 + jsonChunkLen + 8 + binChunkLen;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, 0x46546C67, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  dv.setUint32(12, jsonChunkLen, true);
  dv.setUint32(16, 0x4E4F534A, true);
  out.set(jsonBuf, 20);
  for (let i = 0; i < jsonPad; i++) out[20 + jsonBuf.length + i] = 0x20;
  const binChunkStart = 20 + jsonChunkLen;
  dv.setUint32(binChunkStart, binChunkLen, true);
  dv.setUint32(binChunkStart + 4, 0x004E4942, true);
  out.set(binBytes, binChunkStart + 8);
  return out;
}

function extractGlbJson(glb) {
  const dv = new DataView(glb.buffer, glb.byteOffset, glb.byteLength);
  if (dv.getUint32(0, true) !== 0x46546C67) throw new Error('not a GLB');
  const jsonLen = dv.getUint32(12, true);
  if (dv.getUint32(16, true) !== 0x4E4F534A) throw new Error('expected JSON chunk first');
  const jsonBytes = new Uint8Array(glb.buffer, glb.byteOffset + 20, jsonLen);
  return JSON.parse(new TextDecoder().decode(jsonBytes));
}

async function readSourceGlbExtensions(filePath) {
  const buf = await readFile(filePath);
  const u8 = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (dv.getUint32(0, true) !== 0x46546C67) return { extensions: {}, used: [], required: [] };
  const jsonLen = dv.getUint32(12, true);
  if (dv.getUint32(16, true) !== 0x4E4F534A) return { extensions: {}, used: [], required: [] };
  const json = JSON.parse(new TextDecoder().decode(u8.subarray(20, 20 + jsonLen)));
  return {
    extensions: json.extensions || {},
    used: json.extensionsUsed || [],
    required: json.extensionsRequired || [],
  };
}

// Re-serialize a GLB with a replaced JSON chunk (BIN preserved byte-for-byte).
function rewriteGlbJsonBytes(glbBytes, mutator) {
  const dv = new DataView(glbBytes.buffer, glbBytes.byteOffset, glbBytes.byteLength);
  const jsonLen = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(glbBytes.subarray(20, 20 + jsonLen)));
  mutator(json);
  const newJsonStr = JSON.stringify(json);
  const newJsonBytes = new TextEncoder().encode(newJsonStr);
  const jsonPad = (4 - (newJsonBytes.length % 4)) % 4;
  const newJsonLen = newJsonBytes.length + jsonPad;
  const binChunkStart = 20 + jsonLen;
  const binLen = dv.getUint32(binChunkStart, true);
  const binChunk = glbBytes.subarray(binChunkStart, binChunkStart + 8 + binLen);
  const totalLen = 12 + 8 + newJsonLen + binChunk.byteLength;
  const out = new Uint8Array(totalLen);
  const odv = new DataView(out.buffer);
  odv.setUint32(0, 0x46546C67, true);
  odv.setUint32(4, 2, true);
  odv.setUint32(8, totalLen, true);
  odv.setUint32(12, newJsonLen, true);
  odv.setUint32(16, 0x4E4F534A, true);
  out.set(newJsonBytes, 20);
  for (let i = 0; i < jsonPad; i++) out[20 + newJsonBytes.length + i] = 0x20;
  out.set(binChunk, 20 + newJsonLen);
  return out;
}

// ---------- typed array helpers ----------
function gltfComponentType(typedArray) {
  if (typedArray instanceof Int8Array) return 5120;
  if (typedArray instanceof Uint8Array) return 5121;
  if (typedArray instanceof Int16Array) return 5122;
  if (typedArray instanceof Uint16Array) return 5123;
  if (typedArray instanceof Uint32Array) return 5125;
  if (typedArray instanceof Float32Array) return 5126;
  throw new Error('unknown typed array');
}
function toUint8(typedArray) {
  return new Uint8Array(typedArray.buffer, typedArray.byteOffset, typedArray.byteLength);
}
function typeNumComponents(type) {
  return { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 }[type];
}
const CT_SIZE = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };

// ---------- main ----------
async function main() {
  console.log(`[bake-streaming] input  : ${INPUT}`);
  console.log(`[bake-streaming] output : ${OUT_DIR}`);

  await mkdir(OUT_DIR, { recursive: true });

  await MeshoptSimplifier.ready;
  await MeshoptEncoder.ready;
  await MeshoptDecoder.ready;

  const io = new NodeIO()
    .registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({
      'meshopt.encoder': MeshoptEncoder,
      'meshopt.decoder': MeshoptDecoder,
      'draco3d.decoder': await draco3dgltf.createDecoderModule(),
      'draco3d.encoder': await draco3dgltf.createEncoderModule(),
    });

  // Snapshot passthrough extensions (VRM) from raw source JSON.
  const sourceParts = await readSourceGlbExtensions(INPUT);
  const passthroughBlob = {};
  for (const name of ['VRM']) {
    if (sourceParts.extensions[name]) passthroughBlob[name] = sourceParts.extensions[name];
  }
  if (Object.keys(passthroughBlob).length) {
    console.log(`[bake-streaming] preserving extensions: ${Object.keys(passthroughBlob).join(', ')}`);
  }

  const sourceDoc = await io.read(INPUT);
  const sourceRoot = sourceDoc.getRoot();
  const meshCount = sourceRoot.listMeshes().length;
  const texCount = sourceRoot.listTextures().length;
  console.log(`[bake-streaming] meshes=${meshCount} textures=${texCount}`);
  if (meshCount === 0) {
    console.error('[bake-streaming] input has no meshes — skipping');
    process.exit(2);
  }

  // ---------------- Step 1: build per-primitive LODs ----------------
  // For each LOD, we record raw attribute typed-array bytes + accessor metadata.
  // perPrimLODs: [{ meshIndex, primIndex, lods: [{ ratio, kind, attrs:{POSITION:{type,componentType,count,bytes,min,max,normalized}, ...}, indices:{...}, decodeAABB }] }]
  const perPrimLODs = [];
  const meshes = sourceRoot.listMeshes();

  for (let mi = 0; mi < meshes.length; mi++) {
    const mesh = meshes[mi];
    const prims = mesh.listPrimitives();
    for (let pi = 0; pi < prims.length; pi++) {
      const baselinePrim = prims[pi];
      const baselineVerts = baselinePrim.getAttribute('POSITION')?.getCount() ?? 0;
      const baselineIndices = baselinePrim.getIndices()?.getCount() ?? 0;
      const morphCount = baselinePrim.listTargets()?.length || 0;
      console.log(`[bake-streaming] mesh ${mi} prim ${pi}: ${baselineVerts} verts, ${baselineIndices} indices, ${morphCount} morphs`);

      const lodEntries = [];

      // Decode source baseColor texture for vertex-color stages.
      const matRef = baselinePrim.getMaterial();
      const baseTex = matRef?.getBaseColorTexture?.();
      let baseRGBA = null;
      if (baseTex) {
        const imgBytes = baseTex.getImage();
        if (imgBytes) {
          const decoded = await sharp(Buffer.from(imgBytes)).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
          baseRGBA = { data: decoded.data, width: decoded.info.width, height: decoded.info.height };
        }
      }

      // Helper: given a gltf-transform primitive, extract LOD record bytes.
      function extractFromPrim(simp, ratio, kind) {
        const semantics = simp.listSemantics();
        const attrs = {};
        for (const sem of semantics) {
          const a = simp.getAttribute(sem);
          if (!a) continue;
          const arr = a.getArray();
          attrs[sem] = {
            type: a.getType(),
            componentType: gltfComponentType(arr),
            count: a.getCount(),
            bytes: toUint8(arr).slice(),
            min: a.getMin(new Array(typeNumComponents(a.getType())).fill(0)).slice(),
            max: a.getMax(new Array(typeNumComponents(a.getType())).fill(0)).slice(),
            normalized: a.getNormalized(),
          };
        }
        const idxAcc = simp.getIndices();
        let indices = null;
        if (idxAcc) {
          const arr = idxAcc.getArray();
          indices = {
            componentType: gltfComponentType(arr),
            count: idxAcc.getCount(),
            bytes: toUint8(arr).slice(),
          };
        }
        const posAttr = simp.getAttribute('POSITION');
        const decodeAABB = posAttr ? {
          min: posAttr.getMin(new Array(3).fill(0)).slice(),
          max: posAttr.getMax(new Array(3).fill(0)).slice(),
        } : null;
        return { ratio, kind, semantics, attrs, indices, decodeAABB };
      }

      // Standard textured LODs. For very dense meshes (>500k verts) skip
      // ratio=1.0 — keeping full-density geometry per-LOD bloats output to
      // 50+MB. The 0.4 ratio LOD becomes the HERO tier for these assets.
      const sourcePrim = sourceRoot.listMeshes()[mi].listPrimitives()[pi];
      const posAcc = sourcePrim.getAttribute('POSITION');
      const denseSrc = posAcc && posAcc.getCount() > 500_000;
      const ratios = denseSrc ? MESH_LOD_RATIOS.filter((r) => r < 1.0) : MESH_LOD_RATIOS;
      if (denseSrc) console.log(`[bake-streaming]   dense source (${posAcc.getCount()} verts) — dropping ratio=1.0 LOD`);
      for (const ratio of ratios) {
        const cloneDoc = cloneDocument(sourceDoc);
        const cMesh = cloneDoc.getRoot().listMeshes()[mi];
        cMesh.listPrimitives().forEach((p, idx) => { if (idx !== pi) cMesh.removePrimitive(p); });
        if (ratio < 1.0) {
          await cloneDoc.transform(simplify({ simplifier: MeshoptSimplifier, ratio, error: 0.001, lockBorder: false }));
        }
        const simp = cloneDoc.getRoot().listMeshes()[mi].listPrimitives()[0];
        const rec = extractFromPrim(simp, ratio, 'textured');
        lodEntries.push(rec);
        console.log(`[bake-streaming]   ratio=${ratio} kind=textured idx=${rec.indices?.count ?? 0} verts=${rec.attrs.POSITION?.count ?? 0}`);
      }

      // Extra stages: vertcolor + unskinned.
      for (const stage of EXTRA_LOD_STAGES) {
        const cloneDoc = cloneDocument(sourceDoc);
        const cMesh = cloneDoc.getRoot().listMeshes()[mi];
        cMesh.listPrimitives().forEach((p, idx) => { if (idx !== pi) cMesh.removePrimitive(p); });
        if (stage.kind === 'unskinned') {
          const cmPrim = cMesh.listPrimitives()[0];
          for (const t of cmPrim.listTargets()) cmPrim.removeTarget(t);
        }
        const simplifyError = stage.kind === 'unskinned' ? 0.1 : 0.005;
        await cloneDoc.transform(simplify({ simplifier: MeshoptSimplifier, ratio: stage.ratio, error: simplifyError, lockBorder: false }));
        const simp = cloneDoc.getRoot().listMeshes()[mi].listPrimitives()[0];
        const posAcc = simp.getAttribute('POSITION');
        const vertCount = posAcc?.getCount() ?? 0;

        // Bake vertex colors.
        if (vertCount > 0) {
          const colorArr = buildAveragedVertexColors(simp, baseRGBA, vertCount);
          const colorAccessor = cloneDoc.createAccessor()
            .setType('VEC4')
            .setArray(colorArr)
            .setNormalized(true)
            .setBuffer(cloneDoc.getRoot().listBuffers()[0]);
          simp.setAttribute('COLOR_0', colorAccessor);
        }

        if (stage.kind === 'unskinned') {
          if (simp.getAttribute('JOINTS_0')) simp.setAttribute('JOINTS_0', null);
          if (simp.getAttribute('WEIGHTS_0')) simp.setAttribute('WEIGHTS_0', null);
          for (const t of simp.listTargets()) simp.removeTarget(t);
        }

        const rec = extractFromPrim(simp, stage.ratio, stage.kind);
        lodEntries.push(rec);
        console.log(`[bake-streaming]   ratio=${stage.ratio} kind=${stage.kind} idx=${rec.indices?.count ?? 0} verts=${rec.attrs.POSITION?.count ?? 0}`);
      }

      perPrimLODs.push({ meshIndex: mi, primIndex: pi, lods: lodEntries });
    }
  }

  // ---------------- Step 2: bake texture LOD bytes (webp pyramid) ----------------
  const perTexLODs = [];
  const textures = sourceRoot.listTextures();
  for (let ti = 0; ti < textures.length; ti++) {
    const tex = textures[ti];
    const name = tex.getName() || `tex_${ti}`;
    const img = tex.getImage();
    if (!img) continue;
    const meta = await sharp(Buffer.from(img)).metadata();
    const sizes = TEX_LOD_SIZES.filter((s) => s <= Math.max(meta.width, meta.height));
    if (sizes.length === 0) sizes.push(Math.max(meta.width, meta.height));
    const lods = [];
    for (const sz of sizes) {
      const buf = await sharp(Buffer.from(img))
        .resize(sz, sz, { fit: 'inside', withoutEnlargement: true })
        .webp({ quality: 82 })
        .toBuffer();
      lods.push({ width: sz, bytes: new Uint8Array(buf), mime: 'image/webp' });
    }
    perTexLODs.push({ textureIndex: ti, name, lods });
    console.log(`[bake-streaming] tex ${ti} (${name}): ${lods.length} sizes`);
  }

  // ---------------- Step 3: build final GLB ----------------
  // Round-trip the source through gltf-transform to get baseline JSON (so we
  // get materials, samplers, scene graph etc. in a clean shape). We will then
  // rewrite bufferViews/accessors/buffers/images entirely.
  const baseGlb = await io.writeBinary(sourceDoc);
  const baseJson = extractGlbJson(baseGlb);
  const baseBin = (() => {
    const dv = new DataView(baseGlb.buffer, baseGlb.byteOffset, baseGlb.byteLength);
    const jLen = dv.getUint32(12, true);
    const binChunkStart = 20 + jLen;
    const bLen = dv.getUint32(binChunkStart, true);
    return new Uint8Array(baseGlb.buffer, baseGlb.byteOffset + binChunkStart + 8, bLen);
  })();

  const binParts = [];
  let binCursor = 0;
  const newBufferViews = [];
  const newAccessors = [];

  function pushBytes(bytes, alignment = 4) {
    const pad = (alignment - (binCursor % alignment)) % alignment;
    if (pad) { binParts.push(new Uint8Array(pad)); binCursor += pad; }
    const offset = binCursor;
    binParts.push(bytes);
    binCursor += bytes.byteLength;
    return offset;
  }
  function addBufferView(byteOffset, byteLength, target) {
    const bv = { buffer: 0, byteOffset, byteLength };
    if (target) bv.target = target;
    newBufferViews.push(bv);
    return newBufferViews.length - 1;
  }
  function addAccessor({ bufferView, componentType, count, type, min, max, byteOffset = 0, normalized = false }) {
    const a = { bufferView, componentType, count, type };
    if (byteOffset) a.byteOffset = byteOffset;
    if (normalized) a.normalized = true;
    if (min) a.min = min;
    if (max) a.max = max;
    newAccessors.push(a);
    return newAccessors.length - 1;
  }
  function repackAccessor(oldIndex) {
    const oldAcc = baseJson.accessors?.[oldIndex];
    if (!oldAcc) return -1;
    const oldBv = baseJson.bufferViews?.[oldAcc.bufferView];
    if (!oldBv) return -1;
    const componentSize = CT_SIZE[oldAcc.componentType];
    const numComponents = typeNumComponents(oldAcc.type);
    const elementSize = componentSize * numComponents;
    const sliceStart = (oldBv.byteOffset || 0) + (oldAcc.byteOffset || 0);
    const sliceLen = oldAcc.count * elementSize;
    const slice = baseBin.subarray(sliceStart, sliceStart + sliceLen);
    const newOff = pushBytes(slice, Math.max(4, elementSize));
    const newBv = addBufferView(newOff, slice.byteLength);
    return addAccessor({
      bufferView: newBv,
      componentType: oldAcc.componentType,
      count: oldAcc.count,
      type: oldAcc.type,
      min: oldAcc.min,
      max: oldAcc.max,
      normalized: oldAcc.normalized,
    });
  }

  // Pack each LOD's bufferViews contiguously and record sum byteOffset/byteLength.
  // mesh-LOD records emitted into extras include indicesAcc + attrAccs + decodeAABB.
  const lodMap = [];
  for (const entry of perPrimLODs) {
    const entryRec = { meshIndex: entry.meshIndex, primIndex: entry.primIndex, lods: [] };
    for (const lod of entry.lods) {
      const lodStartOff = binCursor + ((4 - (binCursor % 4)) % 4); // after upcoming alignment
      // First push everything, then compute span.
      const lodRecord = { ratio: lod.ratio, kind: lod.kind, attrAccs: {}, decodeAABB: lod.decodeAABB };
      const spanStart = binCursor; // before pads applied by first pushBytes
      let firstOff = -1;
      if (lod.indices) {
        const off = pushBytes(lod.indices.bytes, 4);
        if (firstOff < 0) firstOff = off;
        const bv = addBufferView(off, lod.indices.bytes.byteLength, 34963);
        lodRecord.indicesAcc = addAccessor({
          bufferView: bv, componentType: lod.indices.componentType, count: lod.indices.count, type: 'SCALAR',
        });
      }
      for (const sem of lod.semantics) {
        const attr = lod.attrs[sem];
        if (!attr) continue;
        const off = pushBytes(attr.bytes, 4);
        if (firstOff < 0) firstOff = off;
        const bv = addBufferView(off, attr.bytes.byteLength, 34962);
        lodRecord.attrAccs[sem] = addAccessor({
          bufferView: bv, componentType: attr.componentType, count: attr.count, type: attr.type,
          min: attr.min, max: attr.max, normalized: attr.normalized,
        });
      }
      const spanEnd = binCursor;
      lodRecord.byteOffset = firstOff >= 0 ? firstOff : spanStart;
      lodRecord.byteLength = spanEnd - lodRecord.byteOffset;
      entryRec.lods.push(lodRecord);
    }
    lodMap.push(entryRec);
  }

  // Texture LODs: pack each as a single bufferView.
  const texMap = [];
  for (const tex of perTexLODs) {
    const rec = { textureIndex: tex.textureIndex, name: tex.name, lods: [] };
    for (const lod of tex.lods) {
      const off = pushBytes(lod.bytes, 4);
      const bv = addBufferView(off, lod.bytes.byteLength);
      rec.lods.push({
        width: lod.width, bufferView: bv, mime: lod.mime,
        byteOffset: off, byteLength: lod.bytes.byteLength,
      });
    }
    texMap.push(rec);
  }

  // Build final JSON: start from baseJson, swap in our new bufferViews / accessors / buffers / images / textures.
  const finalJson = JSON.parse(JSON.stringify(baseJson));
  finalJson.bufferViews = newBufferViews;
  finalJson.accessors = newAccessors;
  finalJson.buffers = [{ byteLength: binCursor }];

  // Decide LOWEST mesh-LOD: prefer 'unskinned' (last entry) — sorted by ratio.
  // Vanilla-loader visible primitive: pick the highest-ratio vertex-only LOD
  // (vertcolor or unskinned). This gives a third-party glTF loader the
  // best-quality vertex-color shape WITHOUT needing any texture, skinning,
  // or compression decoders. Standard loaders see one self-contained mesh.
  // Our streaming runtime ignores this default and routes by extras.
  const pickedDefaultLodByMesh = new Map(); // meshIndex -> kind
  for (const rec of lodMap) {
    const mesh = finalJson.meshes[rec.meshIndex];
    const prim = mesh.primitives[rec.primIndex];
    // Prefer the highest-ratio vertex-only LOD (kind: vertcolor or unskinned).
    // Fall back to the smallest-ratio textured LOD if no vertex-only LOD
    // exists for this primitive.
    const vertexOnly = rec.lods.filter((l) => l.kind === 'vertcolor' || l.kind === 'unskinned');
    let lowest;
    if (vertexOnly.length) {
      vertexOnly.sort((a, b) => b.ratio - a.ratio);
      lowest = vertexOnly[0];
    } else {
      const sorted = [...rec.lods].sort((a, b) => a.ratio - b.ratio);
      lowest = sorted[0];
    }
    pickedDefaultLodByMesh.set(rec.meshIndex, lowest.kind);
    prim.attributes = {};
    for (const sem of Object.keys(lowest.attrAccs)) {
      prim.attributes[sem] = lowest.attrAccs[sem];
    }
    if (lowest.indicesAcc != null) prim.indices = lowest.indicesAcc;
    else delete prim.indices;
    // We've replaced the primitive's bufferViews/accessors with our own
    // (uncompressed) LOD bytes. Strip per-primitive compression extensions
    // copied from baseJson — Draco and meshopt blobs are gone from our BIN.
    if (prim.extensions) {
      delete prim.extensions.KHR_draco_mesh_compression;
      delete prim.extensions.EXT_meshopt_compression;
      if (!Object.keys(prim.extensions).length) delete prim.extensions;
    }
    // If lowest is unskinned, drop morph targets in mesh.primitives.
    if (lowest.kind === 'unskinned') {
      delete prim.targets;
    }
  }

  // Images: one per texture, pointing at the SMALLEST webp LOD.
  finalJson.images = texMap.map((tx) => {
    const smallest = tx.lods[tx.lods.length - 1];
    return { bufferView: smallest.bufferView, mimeType: smallest.mime, name: tx.name };
  });
  finalJson.textures = (finalJson.textures || []).map((t, i) => ({ ...t, source: i }));

  // Repack skin IBM + animation samplers so JSON references valid accessors.
  if (baseJson.skins?.length) {
    finalJson.skins = baseJson.skins.map((s) => {
      const out = { ...s };
      if (s.inverseBindMatrices != null) {
        const ni = repackAccessor(s.inverseBindMatrices);
        if (ni >= 0) out.inverseBindMatrices = ni; else delete out.inverseBindMatrices;
      }
      return out;
    });
  } else {
    delete finalJson.skins;
  }
  if (baseJson.animations?.length) {
    finalJson.animations = baseJson.animations.map((anim) => {
      const out = { ...anim };
      out.samplers = anim.samplers.map((sm) => {
        const ns = { ...sm };
        const ni = repackAccessor(sm.input);
        const no = repackAccessor(sm.output);
        if (ni >= 0) ns.input = ni;
        if (no >= 0) ns.output = no;
        return ns;
      });
      out.channels = anim.channels.map((ch) => ({ ...ch }));
      return out;
    });
  } else {
    delete finalJson.animations;
  }

  // If the picked default LOD is 'unskinned' for a mesh, nodes referencing
  // it should NOT carry a skin reference (the primitive has no
  // JOINTS_0/WEIGHTS_0 in that LOD).
  if (finalJson.nodes) {
    for (const node of finalJson.nodes) {
      if (node.mesh != null && pickedDefaultLodByMesh.get(node.mesh) === 'unskinned' && node.skin != null) {
        delete node.skin;
      }
    }
  }

  // Drop any extensionsUsed/Required that no longer apply (we wrote bare
  // bufferViews — no meshopt/draco extension in lowest LOD). Also drop
  // EXT_texture_webp from required so unknown loaders still parse the doc
  // (they'll just skip texturing; mesh still renders).
  const DROP_USED = new Set(['EXT_meshopt_compression', 'KHR_draco_mesh_compression']);
  const DROP_REQUIRED = new Set(['EXT_meshopt_compression', 'KHR_draco_mesh_compression', 'EXT_texture_webp']);
  finalJson.extensionsUsed = (finalJson.extensionsUsed || []).filter((e) => !DROP_USED.has(e));
  if (finalJson.extensionsRequired) {
    finalJson.extensionsRequired = finalJson.extensionsRequired.filter((e) => !DROP_REQUIRED.has(e));
    if (!finalJson.extensionsRequired.length) delete finalJson.extensionsRequired;
  }
  if (!finalJson.extensionsUsed.length) delete finalJson.extensionsUsed;

  // image/webp requires EXT_texture_webp — standard but extension-gated.
  // To keep "open in any GLTFLoader" honest, only declare it if textures exist.
  if (texMap.length) {
    const used = new Set(finalJson.extensionsUsed || []);
    used.add('EXT_texture_webp');
    finalJson.extensionsUsed = [...used];
    // EXT_texture_webp wraps texture.source via texture.extensions; rather
    // than rewire, we keep the simple texture.source path which most loaders
    // accept for webp images directly (three.js GLTFLoader reads bufferView
    // image and trusts mimeType). Don't mark as required so unknown loaders
    // still parse the document.
  }

  // Attach extras.LOCAL_progressive descriptor.
  finalJson.extras = finalJson.extras || {};
  finalJson.extras.LOCAL_progressive = {
    version: 2,
    streaming: true,
    meshes: lodMap.map((rec) => ({
      meshIndex: rec.meshIndex,
      primIndex: rec.primIndex,
      lods: rec.lods.map((l) => ({
        ratio: l.ratio,
        kind: l.kind,
        byteOffset: l.byteOffset,
        byteLength: l.byteLength,
        indicesAcc: l.indicesAcc,
        attrAccs: l.attrAccs,
        decodeAABB: l.decodeAABB,
      })),
    })),
    textures: texMap.map((rec) => ({
      textureIndex: rec.textureIndex,
      name: rec.name,
      lods: rec.lods.map((l) => ({
        width: l.width,
        bufferView: l.bufferView,
        byteOffset: l.byteOffset,
        byteLength: l.byteLength,
        mime: l.mime,
      })),
    })),
  };

  const binConcat = BufferUtils.concat(binParts);
  let glb = writeGlbBlob(finalJson, binConcat);

  // Splice passthrough extensions (VRM) back into JSON chunk.
  if (Object.keys(passthroughBlob).length) {
    glb = rewriteGlbJsonBytes(glb, (j) => {
      j.extensions = { ...(j.extensions || {}), ...passthroughBlob };
      const used = new Set([...(j.extensionsUsed || []), ...sourceParts.used]);
      j.extensionsUsed = [...used];
      if (sourceParts.required.length) {
        const req = new Set([...(j.extensionsRequired || []), ...sourceParts.required]);
        j.extensionsRequired = [...req];
      }
    });
  }

  const outPath = path.join(OUT_DIR, 'model.progressive.glb');
  await writeFile(outPath, glb);
  const origSize = (await stat(INPUT)).size;
  console.log(`\n[bake-streaming] wrote ${outPath}`);
  console.log(`[bake-streaming] file size : ${(glb.length / 1024 / 1024).toFixed(2)} MB`);
  console.log(`[bake-streaming] orig size : ${(origSize / 1024 / 1024).toFixed(2)} MB`);
}

main().catch((e) => { console.error(e); process.exit(1); });
