#!/usr/bin/env node
// Phase 2 streaming baker.
// Produces a single model.streaming.glb whose binary chunk packs every LOD
// (mesh attributes + texture images) as independent regions referenced by
// glTF bufferViews. The default scene displays the lowest LOD; an extras
// payload `LOCAL_streaming` maps each (meshIndex, primIndex, lodLevel) to
// the exact bufferView indices for indices/POSITION/NORMAL/UV/etc., and
// each texture LOD to a bufferView.
//
// The runtime opens this file with a custom loader that fetches:
//   1. The first 12 bytes (header) + 8 bytes (JSON chunk header) + the JSON
//      chunk via one Range request to learn all bufferView byteOffsets.
//   2. ONLY the bufferViews for the currently-needed LOD via subsequent
//      Range requests. The browser never downloads higher-LOD bytes unless
//      they cross the density threshold.
//
// Because GLB 2.0 allows exactly one BIN chunk per file, we pack all LODs
// into that single chunk and use bufferView byteOffsets to address them.

import { NodeIO, BufferUtils } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { simplify, cloneDocument, prune, dedup } from '@gltf-transform/functions';
import { MeshoptSimplifier } from 'meshoptimizer';
import sharp from 'sharp';
import { mkdir, writeFile, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');

const INPUT = process.argv[2] || path.join(repoRoot, 'model.glb');
const OUT_DIR = process.argv[3] || path.join(repoRoot, 'examples/local-progressive/output');

const MESH_LOD_RATIOS = [1.0, 0.4, 0.15, 0.04];
const TEX_LOD_SIZES = [2048, 1024, 512, 256, 128];

// Helper: write a GLB blob given JSON object + concatenated BIN bytes.
function writeGlbBlob(json, binBytes) {
  const jsonStr = JSON.stringify(json);
  // GLB requires JSON chunk length be multiple of 4 (pad with spaces) and BIN multiple of 4 (pad with zeros).
  const jsonBuf = new TextEncoder().encode(jsonStr);
  const jsonPad = (4 - (jsonBuf.length % 4)) % 4;
  const binPad = (4 - (binBytes.length % 4)) % 4;
  const jsonChunkLen = jsonBuf.length + jsonPad;
  const binChunkLen = binBytes.length + binPad;
  const total = 12 + 8 + jsonChunkLen + 8 + binChunkLen;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  // Header
  dv.setUint32(0, 0x46546C67, true); // 'glTF'
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  // JSON chunk
  dv.setUint32(12, jsonChunkLen, true);
  dv.setUint32(16, 0x4E4F534A, true); // 'JSON'
  out.set(jsonBuf, 20);
  for (let i = 0; i < jsonPad; i++) out[20 + jsonBuf.length + i] = 0x20;
  // BIN chunk
  const binChunkStart = 20 + jsonChunkLen;
  dv.setUint32(binChunkStart, binChunkLen, true);
  dv.setUint32(binChunkStart + 4, 0x004E4942, true); // 'BIN\0'
  out.set(binBytes, binChunkStart + 8);
  // tail already zero
  return out;
}

async function loadAttrBytes(io, doc) {
  // Serialize the doc to a GLB blob and read attributes from there is overkill.
  // Instead, use gltf-transform accessors directly.
  const meshes = doc.getRoot().listMeshes();
  const result = [];
  for (let mi = 0; mi < meshes.length; mi++) {
    const prims = meshes[mi].listPrimitives();
    for (let pi = 0; pi < prims.length; pi++) {
      const prim = prims[pi];
      const entry = { meshIndex: mi, primIndex: pi, attrs: {}, indices: null, semantics: prim.listSemantics() };
      const idx = prim.getIndices();
      if (idx) {
        entry.indices = {
          componentType: gltfComponentType(idx.getArray()),
          count: idx.getCount(),
          bytes: toUint8(idx.getArray()),
        };
      }
      for (const sem of entry.semantics) {
        const a = prim.getAttribute(sem);
        entry.attrs[sem] = {
          type: a.getType(),
          componentType: gltfComponentType(a.getArray()),
          count: a.getCount(),
          bytes: toUint8(a.getArray()),
          min: a.getMin(new Array(typeNumComponents(a.getType())).fill(0)).slice(),
          max: a.getMax(new Array(typeNumComponents(a.getType())).fill(0)).slice(),
        };
      }
      result.push(entry);
    }
  }
  return result;
}

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

async function main() {
  console.log(`[bake-streaming] input  : ${INPUT}`);
  console.log(`[bake-streaming] output : ${OUT_DIR}`);

  await mkdir(OUT_DIR, { recursive: true });
  await MeshoptSimplifier.ready;

  const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
  const sourceDoc = await io.read(INPUT);

  // Step 1: build all LOD attribute byte arrays.
  const perPrimLODs = []; // [{ meshIndex, primIndex, lods: [{ ratio, attrs, indices, semantics }] }]
  const sourcePrims = await loadAttrBytes(io, sourceDoc);

  for (const srcPrim of sourcePrims) {
    const entry = { meshIndex: srcPrim.meshIndex, primIndex: srcPrim.primIndex, lods: [] };
    for (const ratio of MESH_LOD_RATIOS) {
      let lodAttrs, lodIndices, semantics;
      if (ratio === 1.0) {
        lodAttrs = srcPrim.attrs;
        lodIndices = srcPrim.indices;
        semantics = srcPrim.semantics;
      } else {
        const cloneDoc = cloneDocument(sourceDoc);
        const cMesh = cloneDoc.getRoot().listMeshes()[srcPrim.meshIndex];
        const cPrims = cMesh.listPrimitives();
        cPrims.forEach((p, i) => { if (i !== srcPrim.primIndex) cMesh.removePrimitive(p); });
        await cloneDoc.transform(simplify({ simplifier: MeshoptSimplifier, ratio, error: 0.001, lockBorder: false }));
        const decoded = await loadAttrBytes(io, cloneDoc);
        const dPrim = decoded[0];
        lodAttrs = dPrim.attrs;
        lodIndices = dPrim.indices;
        semantics = dPrim.semantics;
      }
      entry.lods.push({ ratio, attrs: lodAttrs, indices: lodIndices, semantics });
      const idxCount = lodIndices?.count ?? 0;
      const vCount = lodAttrs[semantics[0]]?.count ?? 0;
      console.log(`[bake-streaming] mesh ${srcPrim.meshIndex} prim ${srcPrim.primIndex} ratio=${ratio} idx=${idxCount} verts=${vCount}`);
    }
    perPrimLODs.push(entry);
  }

  // Step 2: bake texture LOD bytes.
  const perTexLODs = []; // [{ textureIndex, name, lods: [{ width, bytes, mime }] }]
  const textures = sourceDoc.getRoot().listTextures();
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

  // Step 3: build the GLB JSON + binary, packing every LOD as bufferViews.
  // We reuse most of the source JSON (materials, samplers, scene graph, the
  // original mesh primitive with its material reference) but rewrite the
  // primitive's accessors to point at the LOWEST LOD initially.
  // Higher LODs get extra accessors+bufferViews that aren't referenced by any
  // node — they live in the file purely to be range-fetched.

  // Round-trip the source through gltf-transform to a fresh GLB blob first so
  // we get baseline JSON we can extend.
  const baseGlb = await io.writeBinary(sourceDoc);
  // Parse it back out to JSON + BIN.
  const baseJson = extractGlbJson(baseGlb);

  // Extract the original BIN chunk from baseGlb so we can re-pack referenced
  // ancillary accessors (skin inverse-bind matrices, animation samplers).
  const baseBin = (() => {
    const dv = new DataView(baseGlb.buffer, baseGlb.byteOffset, baseGlb.byteLength);
    const jLen = dv.getUint32(12, true);
    const binChunkStart = 20 + jLen;
    const bLen = dv.getUint32(binChunkStart, true);
    return new Uint8Array(baseGlb.buffer, baseGlb.byteOffset + binChunkStart + 8, bLen);
  })();

  // Build new bufferViews + accessors per-LOD, appending bytes to a single BIN array.
  const binParts = []; // [{ bytes, alignment }]
  let binCursor = 0;
  const newBufferViews = [];
  const newAccessors = [];

  function pushBytes(bytes, alignment = 4) {
    const pad = (alignment - (binCursor % alignment)) % alignment;
    if (pad) {
      binParts.push(new Uint8Array(pad));
      binCursor += pad;
    }
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

  // Repack an accessor from baseJson by copying its bufferView bytes into our
  // new BIN and emitting fresh accessor + bufferView records. Returns the new
  // accessor index, or -1 if the source accessor doesn't exist.
  function repackAccessor(oldIndex) {
    const oldAcc = baseJson.accessors?.[oldIndex];
    if (!oldAcc) return -1;
    const oldBv = baseJson.bufferViews?.[oldAcc.bufferView];
    if (!oldBv) return -1;
    const componentSize = CT_SIZE[oldAcc.componentType];
    const numComponents = typeNumComponents(oldAcc.type);
    const elementSize = componentSize * numComponents;
    // Accessor.byteOffset is offset within the bufferView; bufferView.byteOffset is offset within BIN.
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

  // Pack per-primitive LODs.
  const lodMap = []; // [{ meshIndex, primIndex, lods: [{ ratio, indicesAcc, attrAccs: { POSITION: idx, ... } }] }]
  for (const entry of perPrimLODs) {
    const entryRecord = { meshIndex: entry.meshIndex, primIndex: entry.primIndex, lods: [] };
    for (const lod of entry.lods) {
      const lodRecord = { ratio: lod.ratio, attrAccs: {} };
      if (lod.indices) {
        const off = pushBytes(lod.indices.bytes, 4);
        const bv = addBufferView(off, lod.indices.bytes.byteLength, 34963); // ELEMENT_ARRAY_BUFFER
        lodRecord.indicesAcc = addAccessor({
          bufferView: bv, componentType: lod.indices.componentType, count: lod.indices.count, type: 'SCALAR'
        });
      }
      for (const sem of lod.semantics) {
        const attr = lod.attrs[sem];
        const off = pushBytes(attr.bytes, 4);
        const bv = addBufferView(off, attr.bytes.byteLength, 34962); // ARRAY_BUFFER
        lodRecord.attrAccs[sem] = addAccessor({
          bufferView: bv, componentType: attr.componentType, count: attr.count, type: attr.type, min: attr.min, max: attr.max,
        });
      }
      entryRecord.lods.push(lodRecord);
    }
    lodMap.push(entryRecord);
  }

  // Pack texture LODs.
  const texMap = []; // [{ textureIndex, name, lods: [{ width, bufferView, mime, byteOffset, byteLength }] }]
  for (const tex of perTexLODs) {
    const rec = { textureIndex: tex.textureIndex, name: tex.name, lods: [] };
    for (const lod of tex.lods) {
      const off = pushBytes(lod.bytes, 4);
      const bv = addBufferView(off, lod.bytes.byteLength);
      rec.lods.push({ width: lod.width, bufferView: bv, mime: lod.mime, byteOffset: off, byteLength: lod.bytes.byteLength });
    }
    texMap.push(rec);
  }

  // Build the final glTF JSON.
  // Strategy: start from a minimal subset of baseJson, rewrite mesh primitives
  // to reference the LOWEST LOD's accessors as the initial render-state.
  const finalJson = JSON.parse(JSON.stringify(baseJson));
  finalJson.bufferViews = newBufferViews;
  finalJson.accessors = newAccessors;
  finalJson.buffers = [{ byteLength: binCursor }];

  // Rewrite primitive attribute pointers to lowest LOD (last in our ratios list).
  for (const rec of lodMap) {
    const mesh = finalJson.meshes[rec.meshIndex];
    const prim = mesh.primitives[rec.primIndex];
    const lowest = rec.lods[rec.lods.length - 1];
    prim.attributes = {};
    for (const sem of Object.keys(lowest.attrAccs)) {
      prim.attributes[sem] = lowest.attrAccs[sem];
    }
    if (lowest.indicesAcc != null) prim.indices = lowest.indicesAcc;
  }

  // Rewrite each top-level texture/image to reference the smallest LOD by default.
  finalJson.images = texMap.map((tx) => {
    const smallest = tx.lods[tx.lods.length - 1];
    return {
      bufferView: smallest.bufferView,
      mimeType: smallest.mime,
      name: tx.name,
    };
  });
  // Textures array index map remains as in source; ensure each texture entry references the corresponding image index.
  finalJson.textures = (finalJson.textures || []).map((t, i) => ({ ...t, source: i }));

  // Attach the streaming descriptor.
  finalJson.extras = finalJson.extras || {};
  finalJson.extras.LOCAL_streaming = {
    version: 1,
    meshes: lodMap.map((rec) => ({
      meshIndex: rec.meshIndex,
      primIndex: rec.primIndex,
      lods: rec.lods.map((l) => ({
        ratio: l.ratio,
        indicesAcc: l.indicesAcc,
        attrAccs: l.attrAccs,
      })),
    })),
    textures: texMap.map((rec) => ({
      textureIndex: rec.textureIndex,
      name: rec.name,
      lods: rec.lods.map((l) => ({
        width: l.width,
        bufferView: l.bufferView,
        mime: l.mime,
        byteOffset: l.byteOffset,
        byteLength: l.byteLength,
      })),
    })),
    // Mirror bufferView byte ranges so the runtime doesn't need to re-derive them.
    bufferViewRanges: newBufferViews.map((bv) => ({ byteOffset: bv.byteOffset, byteLength: bv.byteLength })),
  };

  // Re-pack skin inverse-bind matrices (each skin has one IBM accessor)
  // and animation sampler input/output accessors so the new JSON is valid.
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

  // Nodes carrying skin references stay as-is. The mesh primitive's
  // JOINTS_0/WEIGHTS_0 attributes are already wired to lowest-LOD accessors
  // and the skinned-mesh runtime will rebuild them at higher LODs.

  const binConcat = BufferUtils.concat(binParts);
  const glb = writeGlbBlob(finalJson, binConcat);
  const outPath = path.join(OUT_DIR, 'model.streaming.glb');
  await writeFile(outPath, glb);
  const origSize = (await stat(INPUT)).size;
  console.log(`\n[bake-streaming] wrote ${outPath}`);
  console.log(`[bake-streaming] file size : ${(glb.length/1024/1024).toFixed(2)} MB`);
  console.log(`[bake-streaming] orig size : ${(origSize/1024/1024).toFixed(2)} MB`);

  // Compute the byte range needed for the initial fetch: header(12) + json-header(8) + JSON.
  // Also compute total bytes used by lowest-LOD bufferViews to predict initial bandwidth.
  const jsonChunkLen = (() => {
    const dv = new DataView(glb.buffer, glb.byteOffset, glb.byteLength);
    return dv.getUint32(12, true);
  })();
  const jsonEnd = 20 + jsonChunkLen;
  console.log(`[bake-streaming] header+json bytes : ${jsonEnd} (${(jsonEnd/1024).toFixed(1)} KB)`);

  let lowestBytes = 0;
  for (const rec of lodMap) {
    const lowest = rec.lods[rec.lods.length - 1];
    if (lowest.indicesAcc != null) lowestBytes += newBufferViews[newAccessors[lowest.indicesAcc].bufferView].byteLength;
    for (const accIdx of Object.values(lowest.attrAccs)) {
      lowestBytes += newBufferViews[newAccessors[accIdx].bufferView].byteLength;
    }
  }
  for (const rec of texMap) {
    const smallest = rec.lods[rec.lods.length - 1];
    lowestBytes += smallest.byteLength;
  }
  console.log(`[bake-streaming] lowest LOD bytes  : ${lowestBytes} (${(lowestBytes/1024).toFixed(1)} KB)`);
  console.log(`[bake-streaming] minimum first-paint fetch ≈ ${((jsonEnd + lowestBytes)/1024).toFixed(1)} KB`);
}

function extractGlbJson(glb) {
  const dv = new DataView(glb.buffer, glb.byteOffset, glb.byteLength);
  if (dv.getUint32(0, true) !== 0x46546C67) throw new Error('not a GLB');
  const jsonLen = dv.getUint32(12, true);
  const jsonType = dv.getUint32(16, true);
  if (jsonType !== 0x4E4F534A) throw new Error('expected JSON chunk first');
  const jsonBytes = new Uint8Array(glb.buffer, glb.byteOffset + 20, jsonLen);
  return JSON.parse(new TextDecoder().decode(jsonBytes));
}

main().catch((e) => { console.error(e); process.exit(1); });
