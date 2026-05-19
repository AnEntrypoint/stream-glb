#!/usr/bin/env node
// Local progressive LOD baker.
// Takes a GLB, decimates each primitive at multiple ratios using meshoptimizer,
// resizes each texture at multiple sizes via sharp,
// writes a small root GLB carrying the lowest LOD inline plus a LOCAL_progressive
// extension JSON that references sibling .glb / .webp files for higher LODs.

import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { simplify, textureCompress, cloneDocument, prune, dedup, meshopt } from '@gltf-transform/functions';
import { MeshoptSimplifier, MeshoptEncoder, MeshoptDecoder } from 'meshoptimizer';
import draco3dgltf from 'draco3dgltf';
import sharp from 'sharp';
import { mkdir, writeFile, rm, stat, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Read a GLB and return its JSON chunk as a parsed object plus the original
// binary chunk bytes. Used to round-trip extensions gltf-transform doesn't
// know about (e.g. VRM 0.0 — the `VRM` extension is dropped on read).
async function readGlbParts(filePath) {
  const buf = await readFile(filePath);
  const u8 = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (dv.getUint32(0, true) !== 0x46546C67) throw new Error(`${filePath}: not a GLB`);
  const jsonLen = dv.getUint32(12, true);
  if (dv.getUint32(16, true) !== 0x4E4F534A) throw new Error(`${filePath}: expected JSON chunk first`);
  const jsonBytes = u8.subarray(20, 20 + jsonLen);
  const json = JSON.parse(new TextDecoder().decode(jsonBytes));
  return { json, fullBytes: u8 };
}

// Per-channel sRGB → linear lookup. Vertex-color averaging runs in linear
// space so the math matches the renderer's lighting pipeline. We then
// re-encode the averaged linear value back to an sRGB byte for storage;
// the runtime applies a custom onBeforeCompile patch to gamma-decode the
// COLOR_0 attribute at vertex time so it enters the fragment as linear,
// matching how the baseColorTexture is interpreted. Without that, sRGB
// bytes in COLOR_0 get used as linear values directly and the result is
// visibly wrong (typically too saturated and too bright).
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

// Sample a decoded RGBA texture buffer at a UV coordinate, returning the
// per-channel LINEAR value [0,1]. Texture bytes are assumed sRGB-encoded
// (the default for glTF baseColor maps).
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

// Average each surviving vertex's color from the texture by walking the
// SIMPLIFIED mesh's incident triangles and area-weighted-averaging the texel
// samples in LINEAR space (then re-encoding to sRGB for storage). Averaging
// in sRGB space gives visibly wrong perceptual colors — darks too dark and
// midtones shifted. Linear math matches the renderer's lighting pipeline.
function buildAveragedVertexColors(simpPrim, baseRGBA, vertCount) {
  const colorArr = new Uint8Array(vertCount * 4);
  const idxAcc = simpPrim.getIndices();
  const uvAcc = simpPrim.getAttribute('TEXCOORD_0');
  const posAcc = simpPrim.getAttribute('POSITION');
  if (!baseRGBA || !uvAcc) {
    for (let v = 0; v < vertCount; v++) {
      colorArr.set([180, 180, 180, 255], v * 4);
    }
    return colorArr;
  }
  const uv = uvAcc.getArray();
  const pos = posAcc?.getArray();
  // Per-vertex accumulator in LINEAR space: r, g, b, weight.
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
  if (!idx) {
    for (let v = 0; v < vertCount; v += 3) triangleIter(v, v + 1, v + 2);
  } else {
    for (let t = 0; t < idx.length; t += 3) triangleIter(idx[t], idx[t + 1], idx[t + 2]);
  }
  for (let v = 0; v < vertCount; v++) {
    const w = acc[v * 4 + 3];
    if (w > 1e-9) {
      colorArr[v * 4] = linearToSrgbByte(acc[v * 4] / w);
      colorArr[v * 4 + 1] = linearToSrgbByte(acc[v * 4 + 1] / w);
      colorArr[v * 4 + 2] = linearToSrgbByte(acc[v * 4 + 2] / w);
    } else {
      // Isolated vertex: sample once at its own UV (single texel → no average).
      const lin = sampleLinear(baseRGBA.data, baseRGBA.width, baseRGBA.height, uv[v * 2], uv[v * 2 + 1]);
      colorArr[v * 4] = linearToSrgbByte(lin[0]);
      colorArr[v * 4 + 1] = linearToSrgbByte(lin[1]);
      colorArr[v * 4 + 2] = linearToSrgbByte(lin[2]);
    }
    colorArr[v * 4 + 3] = 255;
  }
  return colorArr;
}

// Average one triangle's texel colors in LINEAR space using a barycentric
// sample grid sized to the triangle's UV area. Small triangles (typical
// at high LOD) get a coarse grid; huge triangles (typical at low LOD
// where one survivor covers a big UV region) get many more samples so
// the average reflects the actual texture content.
function triangleAvgColorLinear(uvArr, posArr, ia, ib, ic, baseRGBA) {
  const u0 = uvArr[ia * 2], v0 = uvArr[ia * 2 + 1];
  const u1 = uvArr[ib * 2], v1 = uvArr[ib * 2 + 1];
  const u2 = uvArr[ic * 2], v2 = uvArr[ic * 2 + 1];

  // 3D area for weighting (so big triangles dominate per-vertex average).
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

  // Choose grid resolution N from triangle UV-area in texels: aim for at
  // least 1 sample per ~4 texels so we approximate the true average rather
  // than a sparse spot check. Clamp to [4, 32] so small triangles aren't
  // overworked and huge ones don't blow up bake time.
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
      sr += lin[0];
      sg += lin[1];
      sb += lin[2];
      samples++;
    }
  }
  return { r: sr / samples, g: sg / samples, b: sb / samples, area: area3d };
}

// Re-serialize a GLB with a replaced JSON chunk. The BIN chunk is kept byte-for-byte.
async function rewriteGlbJson(filePath, mutator) {
  const buf = await readFile(filePath);
  const u8 = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const jsonLen = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(u8.subarray(20, 20 + jsonLen)));
  await mutator(json);
  const newJsonStr = JSON.stringify(json);
  const newJsonBytes = new TextEncoder().encode(newJsonStr);
  const jsonPad = (4 - (newJsonBytes.length % 4)) % 4;
  const newJsonLen = newJsonBytes.length + jsonPad;
  const binChunkStart = 20 + jsonLen;
  const binLen = dv.getUint32(binChunkStart, true);
  const binChunk = u8.subarray(binChunkStart, binChunkStart + 8 + binLen); // includes the 8-byte chunk header
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
  await writeFile(filePath, out);
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');

const INPUT = process.argv[2] || path.join(repoRoot, 'model.glb');
// Default output dir is named after the input basename so multiple models
// can be baked into examples/local-progressive/ side by side.
const inputBase = path.basename(INPUT, path.extname(INPUT));
const DEFAULT_OUT = path.join(
  repoRoot,
  'examples/local-progressive',
  inputBase === 'model' ? 'output' : `output_${inputBase}`,
);
const OUT_DIR = process.argv[3] || DEFAULT_OUT;
const LODS_SUBDIR = 'lods';

// LOD recipes from highest to lowest detail. Each entry: { ratio, kind }
//   kind: 'textured'   - standard LOD: simplified mesh, textured material, skinned
//         'vertcolor'  - baked per-vertex colors from baseColor texture, no material map, skinned
//         'unskinned'  - baked per-vertex colors, skin attrs stripped, no longer a SkinnedMesh
// Ratio is the meshopt simplify ratio. The 'ratio' field doubles as the
// sort key in the runtime (ascending = lowest detail first).
const MESH_LOD_RATIOS = [1.0, 0.4, 0.15, 0.04];
const EXTRA_LOD_STAGES = [
  { ratio: 0.04, kind: 'vertcolor' }, // very low, vertex-colored, still skinned
  { ratio: 0.01, kind: 'unskinned' }, // bind-pose, vertex-colored, no skin
];
const TEX_LOD_SIZES = [2048, 1024, 512, 256, 128];

async function main() {
  console.log(`[bake] input  : ${INPUT}`);
  console.log(`[bake] output : ${OUT_DIR}`);

  await rm(OUT_DIR, { recursive: true, force: true });
  await mkdir(path.join(OUT_DIR, LODS_SUBDIR), { recursive: true });

  // Snapshot the source's top-level extensions so we can splice VRM (or any
  // other extension gltf-transform doesn't know) back into the baked output.
  const sourceParts = await readGlbParts(INPUT);
  const sourceExtensions = sourceParts.json.extensions || {};
  const sourceExtensionsUsed = sourceParts.json.extensionsUsed || [];
  const sourceExtensionsRequired = sourceParts.json.extensionsRequired || [];
  const passthroughExtensionNames = ['VRM']; // VRM 0.0
  const passthroughBlob = {};
  for (const name of passthroughExtensionNames) {
    if (sourceExtensions[name]) passthroughBlob[name] = sourceExtensions[name];
  }
  if (Object.keys(passthroughBlob).length) {
    console.log(`[bake] preserving extensions: ${Object.keys(passthroughBlob).join(', ')}`);
  }

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
  const doc = await io.read(INPUT);
  const root = doc.getRoot();

  console.log(`[bake] meshes=${root.listMeshes().length} textures=${root.listTextures().length}`);

  // Stage 1: bake mesh LODs.
  // For each mesh primitive, produce ratios > lowest as sibling .glb chunks.
  // The lowest ratio stays inline in the root document.
  const meshLODs = []; // { meshIndex, primIndex, lods: [{ ratio, path, indexCount, vertexCount, bytes }] }

  const meshes = root.listMeshes();
  for (let mi = 0; mi < meshes.length; mi++) {
    const mesh = meshes[mi];
    const prims = mesh.listPrimitives();
    for (let pi = 0; pi < prims.length; pi++) {
      const baselinePrim = prims[pi];
      const baselineIndices = baselinePrim.getIndices()?.getCount() ?? 0;
      const baselineVerts = baselinePrim.getAttribute('POSITION')?.getCount() ?? 0;
      const morphCount = baselinePrim.listTargets()?.length || 0;
      console.log(`[bake] mesh ${mi} prim ${pi}: ${baselineVerts} verts, ${baselineIndices} indices, ${morphCount} morph targets`);

      const lodEntries = [];

      // gltf-transform's simplify() preserves morph target deltas alongside
      // POSITION/NORMAL when MeshoptSimplifier is used (it runs the simplifier
      // with attribute-aware vertex weighting). Keep mesh LODs even for
      // morph-bearing primitives. The 'unskinned' EXTRA stage still drops
      // morphs along with skin since the bind-pose-frozen target is the
      // whole point of that LOD.
      const skipMeshLod = false;
      const ratios = [...MESH_LOD_RATIOS];
      const lowestRatio = ratios[ratios.length - 1];

      for (const ratio of ratios) {
        const isLowest = ratio === lowestRatio;
        // Clone full document to work on
        const cloneDoc = cloneDocument(doc);
        const cloneMesh = cloneDoc.getRoot().listMeshes()[mi];
        // Replace mesh primitives with only the target one
        const clonePrims = cloneMesh.listPrimitives();
        // Strip other prims
        clonePrims.forEach((p, idx) => { if (idx !== pi) cloneMesh.removePrimitive(p); });

        if (ratio < 1.0) {
          await cloneDoc.transform(
            simplify({ simplifier: MeshoptSimplifier, ratio, error: 0.001, lockBorder: false })
          );
        }

        // Strip everything except the geometry we need (kill textures inside LOD chunks - we ship textures separately)
        const cleanRoot = cloneDoc.getRoot();
        for (const m of cleanRoot.listMaterials()) m.dispose();
        for (const t of cleanRoot.listTextures()) t.dispose();
        for (const m of cleanRoot.listMeshes()) {
          if (m !== cloneMesh) m.dispose();
        }

        const simplified = cloneMesh.listPrimitives()[0];
        const newIndices = simplified?.getIndices()?.getCount() ?? 0;
        const newVerts = simplified?.getAttribute('POSITION')?.getCount() ?? 0;

        if (isLowest) {
          // Lowest LOD will be folded into the root output as the inline geometry.
          // We keep cloneDoc in memory and use it below.
          lodEntries.push({ ratio, inline: true, indexCount: newIndices, vertexCount: newVerts, doc: cloneDoc });
        } else {
          // Capture the pre-quantize POSITION AABB so the runtime can rescale
          // meshopt-decoded vertices (which land in [-1,1]) back into the
          // original character-space coordinates. Without this the swapped
          // geometry renders centered on origin instead of at its true
          // position-space location.
          const posPre = simplified.getAttribute('POSITION');
          const decodeAABB = posPre ? {
            min: posPre.getMin(new Array(3).fill(0)).slice(),
            max: posPre.getMax(new Array(3).fill(0)).slice(),
          } : null;
          // Apply EXT_meshopt_compression on the sibling LOD before serializing
          // — quantizes attributes and encodes for ~5-10x smaller payloads.
          await cloneDoc.transform(
            meshopt({ encoder: MeshoptEncoder, level: 'high' })
          );
          const fileName = `mesh_${mi}_${pi}_r${ratio.toString().replace('.', '')}.glb`;
          const filePath = path.join(OUT_DIR, LODS_SUBDIR, fileName);
          const bin = await io.writeBinary(cloneDoc);
          await writeFile(filePath, bin);
          const sz = (await stat(filePath)).size;
          console.log(`[bake]   ratio=${ratio} -> ${fileName} (${(sz/1024/1024).toFixed(2)} MB, idx=${newIndices})`);
          lodEntries.push({
            ratio,
            path: `${LODS_SUBDIR}/${fileName}`,
            indexCount: newIndices,
            vertexCount: newVerts,
            bytes: sz,
            decodeAABB,
          });
        }
      }

      // Skip extra stages if this primitive carries morph targets — both new
      // stages aggressively decimate, which would shred face blendshapes.
      // SCHWEP3.vrm's face mesh (59 morphs) and LANMOWER mesh 1 (6 morphs)
      // both fall into this bucket and ship only at full detail.
      if (!skipMeshLod) {
        // Decode source baseColor texture once for per-vertex sampling.
        const matRef = baselinePrim.getMaterial();
        const baseTex = matRef?.getBaseColorTexture?.();
        let baseRGBA = null;
        if (baseTex) {
          const imgBytes = baseTex.getImage();
          if (imgBytes) {
            const decoded = await sharp(Buffer.from(imgBytes))
              .ensureAlpha()
              .raw()
              .toBuffer({ resolveWithObject: true });
            baseRGBA = {
              data: decoded.data,
              width: decoded.info.width,
              height: decoded.info.height,
            };
          }
        }

        for (const stage of EXTRA_LOD_STAGES) {
          const cloneDoc = cloneDocument(doc);
          const cloneMesh = cloneDoc.getRoot().listMeshes()[mi];
          const clonePrims = cloneMesh.listPrimitives();
          clonePrims.forEach((p, idx) => { if (idx !== pi) cloneMesh.removePrimitive(p); });
          // For the unskinned stage we want truly aggressive decimation — the
          // mesh ships only at far distances and is rendered via shared
          // InstancedMesh. Drop morph targets up front so the simplifier
          // isn't constrained to preserve their topology, and crank the
          // error tolerance way up so the simplifier doesn't bail early.
          if (stage.kind === 'unskinned') {
            const cmPrim = cloneMesh.listPrimitives()[0];
            for (const t of cmPrim.listTargets()) cmPrim.removeTarget(t);
          }
          const simplifyError = stage.kind === 'unskinned' ? 0.1 : 0.005;
          await cloneDoc.transform(
            simplify({ simplifier: MeshoptSimplifier, ratio: stage.ratio, error: simplifyError, lockBorder: false })
          );

          // Strip materials, textures, sibling meshes — these stages don't
          // sample a texture at render time, so we save bytes.
          const cleanRoot2 = cloneDoc.getRoot();
          for (const m of cleanRoot2.listMaterials()) m.dispose();
          for (const t of cleanRoot2.listTextures()) t.dispose();
          for (const m of cleanRoot2.listMeshes()) {
            if (m !== cloneMesh) m.dispose();
          }

          const simp = cloneMesh.listPrimitives()[0];
          const posAcc = simp.getAttribute('POSITION');
          const uvAcc = simp.getAttribute('TEXCOORD_0');
          const vertCount = posAcc?.getCount() ?? 0;

          // Bake per-vertex colors by area-weighted averaging the texture
          // over each surviving vertex's incident triangles. The 4×4
          // barycentric sample grid acts as a box filter, capturing the
          // mean color of the surface region each vertex now represents
          // after decimation — much closer to the textured appearance than
          // point-sampling a single UV.
          if (vertCount > 0) {
            const colorArr = buildAveragedVertexColors(simp, baseRGBA, vertCount);
            const colorAccessor = cloneDoc
              .createAccessor()
              .setType('VEC4')
              .setArray(colorArr)
              .setNormalized(true)
              .setBuffer(cleanRoot2.listBuffers()[0]);
            simp.setAttribute('COLOR_0', colorAccessor);
          }

          // For the unskinned stage, drop skin attributes, the skin
          // reference, AND morph targets — this LOD is meant for the lowest
          // distance bucket where neither skin animation nor face
          // blendshapes are visible. Done after simplify so the simplifier
          // had full attribute info to make a sane edge-collapse choice.
          if (stage.kind === 'unskinned') {
            const ja = simp.getAttribute('JOINTS_0');
            const wa = simp.getAttribute('WEIGHTS_0');
            if (ja) simp.setAttribute('JOINTS_0', null);
            if (wa) simp.setAttribute('WEIGHTS_0', null);
            // Strip morph targets.
            for (const t of simp.listTargets()) simp.removeTarget(t);
            for (const node of cleanRoot2.listNodes()) {
              if (node.getMesh() === cloneMesh) node.setSkin(null);
            }
            for (const s of cleanRoot2.listSkins()) s.dispose();
          }

          const posPreStage = simp.getAttribute('POSITION');
          const decodeAABB = posPreStage ? {
            min: posPreStage.getMin(new Array(3).fill(0)).slice(),
            max: posPreStage.getMax(new Array(3).fill(0)).slice(),
          } : null;
          await cloneDoc.transform(
            meshopt({ encoder: MeshoptEncoder, level: 'high' })
          );
          const fileName = `mesh_${mi}_${pi}_${stage.kind}.glb`;
          const filePath = path.join(OUT_DIR, LODS_SUBDIR, fileName);
          const bin = await io.writeBinary(cloneDoc);
          await writeFile(filePath, bin);
          const sz = (await stat(filePath)).size;
          console.log(`[bake]   ${stage.kind} r=${stage.ratio} -> ${fileName} (${(sz/1024).toFixed(1)} KB, idx=${simp.getIndices()?.getCount()})`);
          lodEntries.push({
            ratio: stage.ratio,
            kind: stage.kind,
            path: `${LODS_SUBDIR}/${fileName}`,
            indexCount: simp.getIndices()?.getCount() ?? 0,
            vertexCount: vertCount,
            bytes: sz,
            decodeAABB,
          });
        }
      }

      meshLODs.push({ meshIndex: mi, primIndex: pi, lods: lodEntries });
    }
  }

  // Stage 2: bake texture LODs.
  const texLODs = []; // { textureIndex, name, lods: [{ width, path, bytes }] }
  const textures = root.listTextures();
  for (let ti = 0; ti < textures.length; ti++) {
    const tex = textures[ti];
    const name = tex.getName() || `tex_${ti}`;
    const srcImage = tex.getImage();
    if (!srcImage) continue;

    const meta = await sharp(Buffer.from(srcImage)).metadata();
    console.log(`[bake] texture ${ti} (${name}): ${meta.width}x${meta.height} ${meta.format}`);

    // Determine which sizes are <= source. Always include at least the smallest size.
    const sizes = TEX_LOD_SIZES.filter((s) => s <= Math.max(meta.width, meta.height));
    if (sizes.length === 0) sizes.push(Math.max(meta.width, meta.height));
    // Smallest size will be inlined (replaces root texture), bigger sizes are sibling files.
    const lodEntries = [];
    for (const sz of sizes) {
      const buf = await sharp(Buffer.from(srcImage))
        .resize(sz, sz, { fit: 'inside', withoutEnlargement: true })
        .webp({ quality: 82 })
        .toBuffer();
      const isSmallest = sz === sizes[sizes.length - 1];
      if (isSmallest) {
        lodEntries.push({ width: sz, inline: true, bytes: buf.length, buffer: buf });
      } else {
        const fileName = `tex_${ti}_${sz}.webp`;
        const filePath = path.join(OUT_DIR, LODS_SUBDIR, fileName);
        await writeFile(filePath, buf);
        console.log(`[bake]   size=${sz} -> ${fileName} (${(buf.length/1024).toFixed(1)} KB)`);
        lodEntries.push({ width: sz, path: `${LODS_SUBDIR}/${fileName}`, bytes: buf.length });
      }
    }
    texLODs.push({ textureIndex: ti, name, lods: lodEntries });
  }

  // Stage 3: build the root GLB.
  // Strategy: start from the original doc, swap in lowest-LOD geometry per primitive,
  // swap in the smallest texture variant, then attach a top-level LOCAL_progressive
  // extras blob (we use `extras` rather than a registered extension to avoid
  // GLTFLoader rejecting unknown extensionsRequired).
  const rootDoc = cloneDocument(doc);
  const rootRoot = rootDoc.getRoot();

  // Replace primitive indices/attributes with lowest LOD by copying buffer data over.
  for (const ml of meshLODs) {
    const inline = ml.lods.find((x) => x.inline);
    if (!inline) continue;
    const srcPrim = inline.doc.getRoot().listMeshes()[0].listPrimitives()[0];
    const dstPrim = rootRoot.listMeshes()[ml.meshIndex].listPrimitives()[ml.primIndex];

    // Copy indices
    const srcIdx = srcPrim.getIndices();
    if (srcIdx) {
      const newIdx = rootDoc.createAccessor()
        .setType(srcIdx.getType())
        .setArray(srcIdx.getArray().slice())
        .setBuffer(rootRoot.listBuffers()[0]);
      dstPrim.setIndices(newIdx);
    }
    // Copy each attribute
    for (const sem of srcPrim.listSemantics()) {
      const srcAttr = srcPrim.getAttribute(sem);
      if (!srcAttr) continue;
      const newAttr = rootDoc.createAccessor()
        .setType(srcAttr.getType())
        .setArray(srcAttr.getArray().slice())
        .setBuffer(rootRoot.listBuffers()[0]);
      dstPrim.setAttribute(sem, newAttr);
    }
  }

  // Replace textures with smallest WebP.
  for (const tl of texLODs) {
    const tex = rootRoot.listTextures()[tl.textureIndex];
    const inline = tl.lods.find((x) => x.inline);
    if (!inline) continue;
    tex.setImage(inline.buffer);
    tex.setMimeType('image/webp');
  }

  // Build the extension descriptor as extras at the root.
  // Per-primitive density = indexCount / triangle_count_in_world_space_isn't_available_at_bake → use baseline triangle count as proxy.
  const extPayload = {
    version: 1,
    meshes: meshLODs.map((ml) => ({
      meshIndex: ml.meshIndex,
      primIndex: ml.primIndex,
      lods: ml.lods
        .filter((x) => !x.inline)
        .concat([{ ratio: ml.lods.find((x) => x.inline).ratio, inline: true, indexCount: ml.lods.find((x) => x.inline).indexCount }])
        .map((x) => ({
          ratio: x.ratio,
          kind: x.kind || 'textured',
          path: x.path,
          inline: !!x.inline,
          indexCount: x.indexCount,
          vertexCount: x.vertexCount,
          bytes: x.bytes,
          decodeAABB: x.decodeAABB || null,
        })),
    })),
    textures: texLODs.map((tl) => ({
      textureIndex: tl.textureIndex,
      name: tl.name,
      lods: tl.lods.map((x) => ({
        width: x.width,
        path: x.path,
        inline: !!x.inline,
        bytes: x.bytes,
      })),
    })),
  };

  rootRoot.setExtras({ ...rootRoot.getExtras(), LOCAL_progressive: extPayload });

  // Drop orphaned accessors/bufferviews left over after swapping in the lowest LOD geometry.
  await rootDoc.transform(prune(), dedup());

  const rootBin = await io.writeBinary(rootDoc);
  const rootOut = path.join(OUT_DIR, 'model.progressive.glb');
  await writeFile(rootOut, rootBin);

  // Splice the preserved passthrough extensions (VRM etc.) back into the
  // root GLB's JSON chunk. gltf-transform's writer doesn't know about them
  // so it would have dropped them during the round-trip.
  if (Object.keys(passthroughBlob).length) {
    await rewriteGlbJson(rootOut, (j) => {
      j.extensions = { ...(j.extensions || {}), ...passthroughBlob };
      const used = new Set([...(j.extensionsUsed || []), ...sourceExtensionsUsed]);
      j.extensionsUsed = [...used];
      if (sourceExtensionsRequired.length) {
        const req = new Set([...(j.extensionsRequired || []), ...sourceExtensionsRequired]);
        j.extensionsRequired = [...req];
      }
    });
  }
  const rootSize = (await stat(rootOut)).size;
  const origSize = (await stat(INPUT)).size;
  console.log(`\n[bake] root: ${rootOut}`);
  console.log(`[bake] root size : ${(rootSize/1024/1024).toFixed(2)} MB`);
  console.log(`[bake] orig size : ${(origSize/1024/1024).toFixed(2)} MB`);
  console.log(`[bake] saving on initial load: ${(100 - (rootSize/origSize)*100).toFixed(1)}%`);
}

main().catch((e) => { console.error(e); process.exit(1); });
