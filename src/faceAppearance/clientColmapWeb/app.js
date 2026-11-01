import * as THREE from 'three';
import { FaceViewer } from '/viewer-common/scene.js';

const $ = (id) => document.getElementById(id);
const all = (selector) => [...document.querySelectorAll(selector)];
const branchIds = { left: 'legacyTexture', right: 'colmapDirect' };
const state = { manifest: null, selected: null, preview: null, appearance: null, busy: false, syncing: false,
  mode: 'baseColor', frame: null, blobs: new Map(), imageName: 'comparison.png' };
const viewers = {};
const sameRows = (a, b) => a.length === b.length && a.every((row, index) => row.length === b[index].length && row.every((v, c) => v === b[index][c]));

function status(message, error = false) { $('status').textContent = message; $('status').classList.toggle('error', error); }
function controls() {
  for (const id of ['mouth-focus', 'reset-view']) $(id).disabled = state.busy || !state.preview;
  $('case-select').disabled = state.busy || !state.manifest;
  $('mouth-focus').disabled ||= !state.preview?.mouthCenter;
  all('[data-mode], [data-view]').forEach((button) => { button.disabled = state.busy || !state.preview; });
  $('reload').disabled = state.busy;
  $('exposure').disabled = state.busy || !state.preview || state.mode === 'baseColor';
}
function apiPath(url, caseId = null) {
  const parsed = new URL(url, location.origin);
  const prefix = caseId ? `/data/${caseId}/` : '/data/';
  if (parsed.origin !== location.origin || !parsed.pathname.startsWith(prefix)) throw new Error('实验资源地址无效。');
  return parsed.href;
}
async function fetchJSON(url) {
  const response = await fetch(url, { cache: 'no-store', credentials: 'same-origin' });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : `读取实验资源失败（HTTP ${response.status}）。`);
  }
  return response.json();
}
function validateMesh(branch) {
  if (!branch || !Array.isArray(branch.vertices) || !branch.vertices.length
      || !branch.vertices.every((v) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite))
      || !Array.isArray(branch.faces) || !branch.faces.length
      || !branch.faces.every((f) => Array.isArray(f) && f.length === 3 && f.every((v) => Number.isInteger(v) && v >= 0 && v < branch.vertices.length))) {
    throw new Error('实验网格的顶点或三角面无效。');
  }
}
function validateAppearance(appearance, preview, id) {
  if (appearance.schemaVersion !== 1 || (appearance.caseId && appearance.caseId !== id)
      || (appearance.uvOrigin && appearance.uvOrigin !== 'topLeftVDown') || !appearance.branches) throw new Error('贴图说明格式无效。');
  for (const branch of preview.branches) {
    const entry = appearance.branches[branch.id], { uvs, uvFaces } = entry?.mesh || {};
    if (!Array.isArray(uvs) || !uvs.length || !uvs.every((uv) => Array.isArray(uv) && uv.length === 2 && uv.every(Number.isFinite))
        || !Array.isArray(uvFaces) || uvFaces.length !== branch.faces.length
        || !uvFaces.every((face) => Array.isArray(face) && face.length === 3 && face.every((v) => Number.isInteger(v) && v >= 0 && v < uvs.length))) throw new Error('UV 与最终网格不匹配。');
    const texture = entry.texture;
    if (!texture || typeof texture.sha256 !== 'string' || !/^[0-9a-f]{64}$/i.test(texture.sha256)
        || !Number.isSafeInteger(texture.sizeBytes) || texture.sizeBytes <= 0 || texture.colorSpace !== 'sRGB') throw new Error('贴图校验信息无效。');
    apiPath(texture.url, id);
  }
  const [left, right] = Object.values(branchIds).map((key) => appearance.branches[key]);
  if (!sameRows(left.mesh.uvs, right.mesh.uvs) || !sameRows(left.mesh.uvFaces, right.mesh.uvFaces)) throw new Error('两侧客户端 UV 不一致，无法单独对比贴图。');
  return appearance;
}
function textureResolver(descriptor, caseId) {
  return async (url) => {
    const resolved = apiPath(url, caseId);
    if (resolved !== apiPath(descriptor.url, caseId)) throw new Error('贴图地址与说明不一致。');
    const key = `${resolved}|${descriptor.sha256.toLowerCase()}|${descriptor.sizeBytes}`;
    if (!state.blobs.has(key)) state.blobs.set(key, (async () => {
      const response = await fetch(resolved, { cache: 'no-store', credentials: 'same-origin' });
      if (!response.ok) throw new Error(`贴图读取失败（HTTP ${response.status}）。`);
      const bytes = await response.arrayBuffer();
      if (bytes.byteLength !== descriptor.sizeBytes) throw new Error('贴图文件大小校验失败。');
      if (!globalThis.crypto?.subtle) throw new Error('当前浏览器无法校验贴图 SHA-256。');
      const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (v) => v.toString(16).padStart(2, '0')).join('');
      if (hash !== descriptor.sha256.toLowerCase()) throw new Error('贴图 SHA-256 校验失败。');
      return new Blob([bytes], { type: response.headers.get('content-type') || 'image/png' });
    })());
    return state.blobs.get(key);
  };
}
function payload(branch, appearance) {
  // Compute normals before splitting UV seams so only texture coordinates split.
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(branch.vertices.flat(), 3));
  geometry.setIndex(branch.faces.flat());
  geometry.computeVertexNormals();
  const sourceNormals = Array.from(geometry.getAttribute('normal').array);
  geometry.dispose();
  const mesh = { positions: [], normals: [], uvs: [], indices: [], sourceVertexIds: [], groups: [] };
  const pairs = new Map();
  branch.faces.forEach((face, faceId) => face.forEach((vertexId, corner) => {
    const uvId = appearance.mesh.uvFaces[faceId][corner], key = `${vertexId}:${uvId}`;
    let index = pairs.get(key);
    if (index === undefined) {
      index = mesh.sourceVertexIds.length;
      pairs.set(key, index);
      mesh.sourceVertexIds.push(vertexId);
      mesh.positions.push(...branch.vertices[vertexId]);
      mesh.normals.push(...sourceNormals.slice(vertexId * 3, vertexId * 3 + 3));
      mesh.uvs.push(...appearance.mesh.uvs[uvId]);
    }
    mesh.indices.push(index);
  }));
  mesh.groups.push({ start: 0, count: mesh.indices.length, materialIndex: 0 });
  return { templateId: branch.id, coordinates: { lengthUnit: 'centimeter', frame: 'shared final client mesh' }, mesh,
    materials: [{ index: 0, name: branch.id, alphaMode: 'OPAQUE', baseColorFactor: [1, 1, 1, 1], roughnessFactor: 0.85, metallicFactor: 0, texture: appearance.texture }] };
}
function snapshot(viewer) {
  return { position: viewer.camera.position.clone(), quaternion: viewer.camera.quaternion.clone(), up: viewer.camera.up.clone(),
    target: viewer.controls.target.clone(), zoom: viewer.camera.zoom, center: viewer.center.clone(), fitDistance: viewer.fitDistance };
}
function restore(viewer, pose) {
  viewer.camera.position.copy(pose.position); viewer.camera.quaternion.copy(pose.quaternion); viewer.camera.up.copy(pose.up);
  viewer.camera.zoom = pose.zoom; viewer.center.copy(pose.center); viewer.fitDistance = pose.fitDistance;
  viewer.controls.target.copy(pose.target); viewer.camera.updateProjectionMatrix(); viewer.controls.update();
}
function synchronize(from, to) {
  if (state.busy || state.syncing) return;
  state.syncing = true;
  try { restore(to, snapshot(from)); } finally { state.syncing = false; }
}
try {
  for (const side of ['left', 'right']) {
    viewers[side] = new FaceViewer($(`${side}-viewport`), { onError: (error) => status(error.message, true) });
    viewers[side].controls.enableDamping = false;
  }
  viewers.left.controls.addEventListener('change', () => synchronize(viewers.left, viewers.right));
  viewers.right.controls.addEventListener('change', () => synchronize(viewers.right, viewers.left));
} catch (error) { status(`无法启用三维预览：${error.message}。请开启浏览器硬件加速。`, true); }
function applyMode(viewer) {
  viewer.setMode(state.mode === 'baseColor' ? 'textured' : state.mode);
  if (state.mode === 'baseColor' && viewer.asset) {
    const asset = viewer.asset;
    if (!asset.modes.baseColor) {
      asset.modes.baseColor = asset.modes.textured.map((material) => new THREE.MeshBasicMaterial({
        name: `${material.name}:baseColor`, color: 0xffffff, map: material.map, transparent: false, side: THREE.FrontSide, toneMapped: false,
      }));
      asset.materials.push(...asset.modes.baseColor);
    }
    asset.mesh.material = asset.modes.baseColor;
  }
}
function frameMesh(branch) {
  const box = new THREE.Box3();
  branch.vertices.forEach((v) => box.expandByPoint(new THREE.Vector3(...v)));
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  state.frame = { center: sphere.center, radius: Math.max(sphere.radius, 0.001) };
}
function applyFrame() {
  const { center, radius } = state.frame;
  for (const viewer of Object.values(viewers)) {
    viewer.center.copy(center); viewer.radius = radius; viewer.hasBounds = true;
    viewer.lightRig.position.copy(center); viewer.lightRig.scale.setScalar(radius);
    viewer.camera.near = radius / 1000; viewer.camera.far = radius * 100;
    viewer.controls.minDistance = radius * 0.08; viewer.controls.maxDistance = radius * 15;
    viewer.camera.updateProjectionMatrix();
  }
  const distance = Math.max(...Object.values(viewers).map((v) => v._fitDistance()));
  Object.values(viewers).forEach((viewer) => { viewer.fitDistance = distance; });
}
function setView(view) {
  if (!state.preview || state.busy) return;
  state.syncing = true;
  try { viewers.left.setView(view); restore(viewers.right, snapshot(viewers.left)); } finally { state.syncing = false; }
}
function focus(mouth) {
  if (!state.preview || state.busy) return;
  state.syncing = true;
  try {
    applyFrame();
    if (mouth && state.preview.mouthCenter) {
      const center = new THREE.Vector3(...state.preview.mouthCenter);
      // Keep resize() fitted to the inspected region as well as the first frame.
      for (const viewer of Object.values(viewers)) viewer.radius = state.frame.radius * 0.25;
      const distance = Math.max(...Object.values(viewers).map((viewer) => viewer._fitDistance()));
      for (const viewer of Object.values(viewers)) { viewer.center.copy(center); viewer.fitDistance = distance; }
    }
    viewers.left.setView('front'); restore(viewers.right, snapshot(viewers.left));
  } finally { state.syncing = false; }
}
function metrics(side, branch, appearance) {
  const container = $(`${side}-metrics`);
  container.replaceChildren();
  for (const [label, value] of [['顶点', branch.vertices.length.toLocaleString('zh-CN')], ['三角面', branch.faces.length.toLocaleString('zh-CN')], ['贴图 SHA-256', appearance.texture.sha256.slice(0, 12)]]) {
    const item = document.createElement('div'), title = document.createElement('span'), number = document.createElement('b');
    item.className = 'metric'; title.textContent = label; number.textContent = value; item.append(title, number); container.append(item);
  }
  $(`${side}-texture-tag`).textContent = appearance.provenanceLabel || (side === 'left' ? '复用原客户端贴图' : '照片 → 最终网格 → 客户端 UV');
}
function showEvidence(name = state.imageName) {
  if (!state.selected) return;
  const descriptions = { 'comparison.png': '原照片与同一最终网格的两种贴图，在固定正面相机下对照。点击查看原大小。',
    'mouthComparison.png': '固定正面相机的嘴部放大，用于检查唇线、嘴角和贴图边界。点击查看原大小。',
    'sideComparison.png': '同一几何与侧面视角，比较新旧贴图的连续性。点击查看原大小。',
    'coverage.png': '客户端 UV 中的照片覆盖：白色为通过来源遮罩检查的烘焙色，黑色为没有照片支持。包含后脑和颈部，不能当作脸部覆盖率。点击查看原大小。' };
  if (!Object.hasOwn(descriptions, name)) return;
  state.imageName = name;
  const image = $('evidence-image'), url = apiPath(`/data/${state.selected}/${name}`, state.selected);
  image.hidden = true; $('image-status').hidden = false; $('image-status').textContent = '正在读取图片…';
  $('image-description').textContent = descriptions[name]; $('image-full').href = url;
  image.alt = `${$('case-title').textContent} · ${descriptions[name]}`;
  image.onload = () => { if (image.src === url) { image.hidden = false; $('image-status').hidden = true; } };
  image.onerror = () => { if (image.src === url) { image.hidden = true; $('image-status').hidden = false; $('image-status').textContent = '此对照图尚未生成。'; } };
  image.src = url;
  all('[data-image]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.image === name)));
}
async function chooseCase(id) {
  if (state.busy) return;
  const entry = state.manifest?.cases.find((candidate) => candidate.id === id);
  if (!entry || !viewers.left || !viewers.right) return;
  state.busy = true; controls(); status('正在读取最终几何与两组贴图…');
  all('.viewport').forEach((element) => { element.style.visibility = 'hidden'; });
  try {
    const preview = await fetchJSON(apiPath(entry.previewUrl || `/data/${id}/preview.json`, id));
    if (preview.caseId !== id || !Array.isArray(preview.branches)) throw new Error('实验预览格式无效。');
    const selected = Object.fromEntries(Object.entries(branchIds).map(([side, branchId]) => [side, preview.branches.find((branch) => branch.id === branchId)]));
    Object.values(selected).forEach(validateMesh);
    if (!sameRows(selected.left.vertices, selected.right.vertices) || !sameRows(selected.left.faces, selected.right.faces)) throw new Error('两侧几何不一致，无法单独对比贴图。');
    if (preview.mouthCenter && (!Array.isArray(preview.mouthCenter) || preview.mouthCenter.length !== 3 || !preview.mouthCenter.every(Number.isFinite))) throw new Error('嘴部观察中心无效。');
    const appearance = validateAppearance(await fetchJSON(apiPath(`/data/${id}/appearance.json`, id)), preview, id);
    state.blobs.clear();
    const loads = await Promise.allSettled(['left', 'right'].map((side) => viewers[side].load(payload(selected[side], appearance.branches[branchIds[side]]), textureResolver(appearance.branches[branchIds[side]].texture, id))));
    const failure = loads.find((result) => result.status === 'rejected');
    if (failure) throw failure.reason;
    state.selected = id; state.preview = preview; state.appearance = appearance; frameMesh(selected.left); applyFrame();
    viewers.left.setView('front'); restore(viewers.right, snapshot(viewers.left));
    for (const side of ['left', 'right']) { applyMode(viewers[side]); metrics(side, selected[side], appearance.branches[branchIds[side]]); }
    all('.viewport').forEach((element) => { element.style.visibility = 'visible'; });
    const textureDifferent = appearance.branches.legacyTexture.texture.sha256.toLowerCase() !== appearance.branches.colmapDirect.texture.sha256.toLowerCase();
    $('appearance-status').textContent = textureDifferent ? '已校验：两侧几何与客户端 UV 完全一致，使用不同贴图。' : '两侧几何、UV 和贴图均相同；尚无新的烘焙差异。';
    $('appearance-status').classList.toggle('fallback', !textureDifferent);
    $('case-select').value = id; $('case-title').textContent = entry.label || id;
    for (const [element, filename] of Object.entries({ 'old-texture': 'oldTexture.png', 'new-texture': 'colmapTexture.png', 'report-link': 'report.json' })) $(element).href = apiPath(`/data/${id}/${filename}`, id);
    showEvidence(); status('两侧镜头同步 · 几何固定，仅比较贴图取色方式。');
    const url = new URL(location.href); url.searchParams.set('case', id); history.replaceState(null, '', url);
  } catch (error) {
    state.preview = null; status(error.message, true);
    $('appearance-status').textContent = '本组对照未完成载入，请刷新结果。'; $('appearance-status').classList.add('fallback');
  } finally { state.busy = false; controls(); $('mouth-focus').disabled ||= !state.preview?.mouthCenter; }
}
async function loadManifest() {
  if (state.busy) return;
  state.busy = true; controls();
  try {
    const manifest = await fetchJSON('/api/experiment');
    if (!Array.isArray(manifest.cases) || !manifest.cases.length || !manifest.cases.every((entry) => typeof entry.id === 'string' && /^[A-Za-z0-9_-]{1,100}$/.test(entry.id))) throw new Error('尚未生成可预览的实验组，请稍后刷新。');
    state.manifest = manifest; $('case-select').replaceChildren();
    manifest.cases.forEach((entry) => $('case-select').add(new Option(entry.label || entry.id, entry.id)));
  } catch (error) { status(error.message, true); }
  finally { state.busy = false; controls(); }
  if (state.manifest?.cases.length) {
    const wanted = state.selected || new URL(location.href).searchParams.get('case');
    await chooseCase(state.manifest.cases.some((entry) => entry.id === wanted) ? wanted : state.manifest.cases[0].id);
  }
}
$('case-select').addEventListener('change', (event) => chooseCase(event.target.value));
$('reload').addEventListener('click', loadManifest);
all('[data-image]').forEach((button) => button.addEventListener('click', () => showEvidence(button.dataset.image)));
all('[data-mode]').forEach((button) => button.addEventListener('click', () => {
  state.mode = button.dataset.mode; Object.values(viewers).forEach(applyMode);
  all('[data-mode]').forEach((entry) => entry.setAttribute('aria-pressed', String(entry === button))); controls();
}));
all('[data-view]').forEach((button) => button.addEventListener('click', () => setView(button.dataset.view)));
$('mouth-focus').addEventListener('click', () => focus(true));
$('reset-view').addEventListener('click', () => focus(false));
$('exposure').addEventListener('input', (event) => {
  const value = Number(event.target.value); $('exposure-value').textContent = value.toFixed(2);
  Object.values(viewers).forEach((viewer) => viewer.setExposure(value));
});
window.addEventListener('pagehide', () => Object.values(viewers).forEach((viewer) => viewer.dispose()), { once: true });
controls();
await loadManifest();
