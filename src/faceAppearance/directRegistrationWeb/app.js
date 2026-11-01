import * as THREE from 'three';
import { FaceViewer } from '/viewer-common/scene.js';

const $ = (id) => document.getElementById(id);
const all = (selector) => [...document.querySelectorAll(selector)];
const branchNames = { source: '原始 UFF · 参考', baseline: 'A · 当前 UV 搬运', legacyPrior: 'B · 旧先验 1000', weakPrior: 'C · 弱先验 100', noPrior: 'D · 无形状先验',
  noPriorHard: 'E · 无先验 / 固定接口', noPriorHardSmooth300: 'F · 固定接口 / 较弱平滑' };
const caseNames = { maleExactRegression: '歪嘴回归 · 男性模板', orbitControl4: '环绕基线 · 女性模板', highPitch4: '明显抬头 · 扭头组', cyanWall4: '浅蓝背景 · 低清扭头组', outdoorSideLight4: '室外方向光 · 低清扭头组', backlitCap4: '逆光帽檐 · 挑战组' };
const state = { manifest: null, safety: null, preview: null, selected: null, busy: false, generation: 0, mode: 'clay', frame: null, focus: 'face', syncing: false };
const viewers = {};

function status(text, error = false) { $('status').textContent = text; $('status').classList.toggle('error', error); }
function controls() {
  for (const id of ['case-select', 'left-branch', 'right-branch', 'mouth-focus', 'reset-view', 'exposure']) $(id).disabled = state.busy || !state.preview;
  all('[data-mode], [data-view]').forEach((button) => { button.disabled = state.busy || !state.preview; });
  $('reload').disabled = state.busy;
}
function apiPath(url) {
  const parsed = new URL(url, location.origin);
  if (parsed.origin !== location.origin || !parsed.pathname.startsWith('/data/')) throw new Error('实验资源地址无效。');
  return parsed.href;
}
async function fetchJSON(path) {
  const response = await fetch(path, { cache: 'no-store', credentials: 'same-origin' });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : `无法读取实验数据（HTTP ${response.status}）。`);
  }
  return response.json();
}

function snapshot(viewer) {
  return { position: viewer.camera.position.clone(), quaternion: viewer.camera.quaternion.clone(), up: viewer.camera.up.clone(),
    target: viewer.controls.target.clone(), zoom: viewer.camera.zoom, center: viewer.center.clone(), fitDistance: viewer.fitDistance };
}
function restore(viewer, pose) {
  viewer.camera.position.copy(pose.position);
  viewer.camera.quaternion.copy(pose.quaternion);
  viewer.camera.up.copy(pose.up);
  viewer.camera.zoom = pose.zoom;
  viewer.center.copy(pose.center);
  viewer.fitDistance = pose.fitDistance;
  viewer.controls.target.copy(pose.target);
  viewer.camera.updateProjectionMatrix();
  viewer.controls.update();
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
    viewers[side].setMode('clay');
  }
  viewers.left.controls.addEventListener('change', () => synchronize(viewers.left, viewers.right));
  viewers.right.controls.addEventListener('change', () => synchronize(viewers.right, viewers.left));
} catch (error) {
  status(`无法启用三维预览：${error.message}。请开启浏览器硬件加速。`, true);
}

function branchPayload(branch) {
  const { vertices, faces } = branch;
  if (!Array.isArray(vertices) || !vertices.length || !vertices.every((v) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite))
      || !Array.isArray(faces) || !faces.length || !faces.every((f) => Array.isArray(f) && f.length === 3 && f.every((i) => Number.isInteger(i) && i >= 0 && i < vertices.length))) {
    throw new Error('实验网格的顶点或三角面无效。');
  }
  const positions = vertices.flat(), indices = faces.flat();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  const normals = Array.from(geometry.getAttribute('normal').array);
  geometry.dispose();
  return { templateId: branch.id, coordinates: { lengthUnit: 'centimeter', frame: 'shared baseline template' },
    mesh: { positions, normals, indices, uvs: Array(vertices.length * 2).fill(0), sourceVertexIds: vertices.map((_, index) => index),
      groups: [{ start: 0, count: indices.length, materialIndex: 0 }] },
    materials: [{ index: 0, name: branch.id, alphaMode: 'OPAQUE', baseColorFactor: [0.6, 0.7, 0.63, 1], roughnessFactor: 0.85, metallicFactor: 0 }] };
}

function commonFrame(preview) {
  const bounds = new THREE.Box3();
  for (const branch of preview.branches) for (const point of branch.vertices) bounds.expandByPoint(new THREE.Vector3(...point));
  const sphere = bounds.getBoundingSphere(new THREE.Sphere());
  return { center: sphere.center, radius: Math.max(sphere.radius, 0.001) };
}

function applyFrame() {
  const { center, radius } = state.frame;
  for (const viewer of Object.values(viewers)) {
    viewer.center.copy(center);
    viewer.radius = radius;
    viewer.hasBounds = true;
    viewer.lightRig.position.copy(center);
    viewer.lightRig.scale.setScalar(radius);
    viewer.camera.near = radius / 1000;
    viewer.camera.far = radius * 100;
    viewer.controls.minDistance = radius * 0.08;
    viewer.controls.maxDistance = radius * 15;
    viewer.camera.updateProjectionMatrix();
  }
  const fitDistance = Math.max(...Object.values(viewers).map((viewer) => viewer._fitDistance()));
  for (const viewer of Object.values(viewers)) viewer.fitDistance = fitDistance;
}

function setView(view) {
  if (!state.preview || state.busy) return;
  state.syncing = true;
  try {
    viewers.left.setView(view);
    restore(viewers.right, snapshot(viewers.left));
  } finally { state.syncing = false; }
}

function focus(region) {
  if (!state.preview || state.busy) return;
  state.focus = region;
  state.syncing = true;
  try {
    applyFrame();
    if (region === 'mouth') {
      const center = new THREE.Vector3(...state.preview.mouthCenter);
      const distance = Math.max(...Object.values(viewers).map((viewer) => {
        const halfVertical = THREE.MathUtils.degToRad(viewer.camera.fov / 2);
        const halfHorizontal = Math.atan(Math.tan(halfVertical) * viewer.camera.aspect);
        return 4.2 / Math.sin(Math.min(halfVertical, halfHorizontal));
      }));
      for (const viewer of Object.values(viewers)) { viewer.center.copy(center); viewer.fitDistance = distance; }
    }
    viewers.left.setView('front');
    restore(viewers.right, snapshot(viewers.left));
  } finally { state.syncing = false; }
}

function metric(container, name, value, bad = false) {
  const item = document.createElement('div');
  item.className = 'metric';
  const title = document.createElement('span');
  title.textContent = name;
  const number = document.createElement('b');
  number.textContent = value;
  number.classList.toggle('bad', bad);
  item.append(title, number);
  container.append(item);
}
function renderMetrics(side, branch) {
  const container = $(`${side}-metrics`), metrics = branch.metrics || {}, quality = metrics.quality;
  const audit = state.safety?.cases.find((entry) => entry.caseId === state.selected);
  const intersection = audit?.meshMatchesAudit ? audit.branches[branch.id] : null;
  container.replaceChildren();
  $(`${side}-tag`).textContent = branch.id === 'source' ? '原生 UFF · 参考' : branch.id === 'baseline' ? '当前方法 · 对照' : '本地实验 · 候选';
  if (!quality) {
    metric(container, '顶点', branch.vertices.length.toLocaleString('zh-CN'));
    metric(container, '三角面', branch.faces.length.toLocaleString('zh-CN'));
    const note = document.createElement('span');
    note.className = 'metric-status';
    note.textContent = '原生输入参考；源网格也存在唇部等区域的自交，详见审计报告。其拓扑不同，不能与客户端直接比较交叠对数。';
    container.append(note);
    return;
  }
  if (intersection) {
    const count = intersection.summary.positiveLengthOrAreaPairs;
    metric(container, '非邻接面穿插 / 交叠对', String(count), count > 0);
    metric(container, '相对 A 新增对', String(intersection.newPositivePairsVsBaseline), intersection.newPositivePairsVsBaseline > 0);
    metric(container, '相对原模板新增对', String(intersection.newPositivePairsVsRest), intersection.newPositivePairsVsRest > 0);
  } else metric(container, '穿插审计', '未载入', true);
  metric(container, '相对模板翻面', String(quality.invertedFaceCount ?? '—'), quality.invertedFaceCount > 0);
  metric(container, '嘴部相对翻面', String(quality.mouthInvertedFaceCount ?? '—'), quality.mouthInvertedFaceCount > 0);
  metric(container, '退化面', String(quality.collapsedTriangleCount ?? '—'), quality.collapsedTriangleCount > 0);
  if (Number.isFinite(metrics.staticDriftMaxMm)) metric(container, '静态点漂移 / mm', metrics.staticDriftMaxMm.toFixed(3));
  if (Number.isFinite(metrics.elapsedSeconds)) metric(container, '配准 / 秒', metrics.elapsedSeconds.toFixed(3));
  if (Number.isFinite(metrics.guardedVertexCount)) metric(container, '回退保护顶点', String(metrics.guardedVertexCount));
  if (intersection) {
    const totals = document.createElement('span');
    totals.className = 'metric-intersection';
    const rest = audit.branches.rest?.summary.positiveLengthOrAreaPairs ?? '—';
    const baseline = audit.branches.baseline?.summary.positiveLengthOrAreaPairs ?? '—';
    totals.textContent = `原模板 ${rest} 对 · A 当前方法 ${baseline} 对 · 本分支 ${intersection.summary.positiveLengthOrAreaPairs} 对。相对 A 移除 ${intersection.removedPositivePairsVsBaseline} 对，同时新增 ${intersection.newPositivePairsVsBaseline} 对。`;
    container.append(totals);
  }
  const note = document.createElement('span');
  note.className = 'metric-status failed';
  note.textContent = branch.id === 'baseline' ? '当前方法也存在已知交叠；相对翻面计数不代表无自交。'
    : '本地候选 · 本轮未达到部署标准，不以法向或观感单独判定稳定。';
  if (audit && !audit.meshMatchesAudit) note.textContent = '审计记录与当前几何不匹配，请重新审计；本分支尚未获稳定验收。';
  container.append(note);
}

async function showBranches(keepView = true) {
  if (!state.preview || state.busy || !viewers.left || !viewers.right) return;
  const selected = Object.fromEntries(['left', 'right'].map((side) => [side, state.preview.branches.find((branch) => branch.id === $(`${side}-branch`).value)]));
  if (!selected.left || !selected.right) return;
  const pose = keepView ? snapshot(viewers.left) : null;
  state.busy = true;
  controls();
  status('正在载入两侧几何…');
  try {
    await Promise.all(['left', 'right'].map((side) => viewers[side].load(branchPayload(selected[side]), async () => { throw new Error('此实验没有纹理。'); })));
    applyFrame();
    if (pose) {
      for (const viewer of Object.values(viewers)) restore(viewer, pose);
    } else {
      viewers.left.setView('front');
      restore(viewers.right, snapshot(viewers.left));
      state.focus = 'face';
    }
    for (const side of ['left', 'right']) { viewers[side].setMode(state.mode); renderMetrics(side, selected[side]); }
    status('同一基线坐标 · 两侧相机同步 · 本轮候选尚未达到部署标准，未替换远程服务。');
    const url = new URL(location.href);
    url.searchParams.set('case', state.selected);
    url.searchParams.set('left', selected.left.id);
    url.searchParams.set('right', selected.right.id);
    history.replaceState(null, '', url);
  } catch (error) { status(error.message, true); }
  finally { state.busy = false; controls(); }
}

async function chooseCase(id, preferences = {}) {
  if (state.busy) return;
  const entry = state.manifest?.cases.find((candidate) => candidate.id === id);
  if (!entry) return;
  state.busy = true;
  const generation = ++state.generation;
  controls();
  status('正在读取此组的全部配准分支…');
  try {
    const preview = await fetchJSON(apiPath(entry.previewUrl));
    if (generation !== state.generation) return;
    if (preview.caseId !== id || !Array.isArray(preview.branches) || !preview.branches.length
        || !Array.isArray(preview.mouthCenter) || preview.mouthCenter.length !== 3 || !preview.mouthCenter.every(Number.isFinite)) throw new Error('实验预览格式无效。');
    // Validate every branch before including it in the shared bounds.
    preview.branches.forEach(branchPayload);
    state.preview = preview;
    state.selected = id;
    state.frame = commonFrame(preview);
    $('case-select').value = id;
    for (const side of ['left', 'right']) {
      const select = $(`${side}-branch`);
      const wanted = preferences[side] || select.value || (side === 'left' ? 'source' : 'noPriorHardSmooth300');
      select.replaceChildren();
      for (const branch of preview.branches) select.add(new Option(branchNames[branch.id] || branch.label, branch.id));
      select.value = preview.branches.some((branch) => branch.id === wanted) ? wanted
        : side === 'right' && preview.branches.some((branch) => branch.id === 'noPrior') ? 'noPrior' : preview.branches[0].id;
    }
    $('case-title').textContent = caseNames[id] || id;
    const capture = entry.captureMode === 'userConfirmedCameraOrbit' ? '相机环绕（用户已确认）' : '该批次主要扭头，未逐目录标定';
    $('case-description').textContent = `${entry.gender === 'male' ? '男性' : '女性'}实验模板 · ${capture}。不同目录代表不同采集组，不保证独立身份。`;
    $('mouth-image').href = apiPath(`/data/${id}/mouthSelected.png`);
    $('mouth-all').href = apiPath(entry.mouthImage || `/data/${id}/mouth.png`);
    $('face-image').href = apiPath(`/data/${id}/face.png`);
    $('mesh-download').href = apiPath(`/data/${id}/meshes.npz`);
  } catch (error) { status(error.message, true); state.preview = null; }
  finally { state.busy = false; controls(); }
  if (state.preview && generation === state.generation) await showBranches(false);
}

async function loadManifest() {
  if (state.busy) return;
  state.busy = true;
  controls();
  try {
    const [manifestResult, safetyResult] = await Promise.allSettled([fetchJSON('/api/experiment'), fetchJSON('/api/safety')]);
    if (manifestResult.status !== 'fulfilled') throw manifestResult.reason;
    const manifest = manifestResult.value;
    state.safety = safetyResult.status === 'fulfilled' ? safetyResult.value : null;
    if (state.safety?.cases?.length) {
      const invalid = state.safety.cases.some((entry) => !entry.meshMatchesAudit);
      $('audit-summary').textContent = invalid ? '部分审计记录与当前几何不匹配。刷新审计前不判定稳定，所有候选仍保留在本地。'
        : `三角面求交审计：F 在 ${state.safety.fIntersectionIncreaseCaseCount} / ${state.safety.caseCount} 组中的穿插 / 交叠总数高于 A。各分支均需检查新增交叠位置。`;
    } else $('audit-summary').textContent = '交叠审计尚未载入。所有候选保留在本地，不将法向翻转计数视为自交检查。';
    if (!Array.isArray(manifest.cases) || !manifest.cases.length) throw new Error('尚未生成可预览的实验组，请稍后刷新。');
    state.manifest = manifest;
    $('case-select').replaceChildren();
    manifest.cases.forEach((entry) => $('case-select').add(new Option(caseNames[entry.id] || entry.id, entry.id)));
    $('objective').textContent = `实验目标：${manifest.objective || '见实验报告'}`;
    $('solver').textContent = `求解方式：${manifest.solver || '见实验报告'}。不将目标函数相同等同于原优化迭代轨迹相同。`;
  } catch (error) { status(error.message, true); }
  finally { state.busy = false; controls(); }
  if (state.manifest?.cases.length) {
    const query = new URL(location.href).searchParams;
    const wanted = state.selected || query.get('case');
    await chooseCase(state.manifest.cases.some((entry) => entry.id === wanted) ? wanted : state.manifest.cases[0].id,
      { left: query.get('left') || 'source', right: query.get('right') || 'noPriorHardSmooth300' });
  }
}

$('case-select').addEventListener('change', (event) => chooseCase(event.target.value));
$('left-branch').addEventListener('change', () => showBranches());
$('right-branch').addEventListener('change', () => showBranches());
$('reload').addEventListener('click', loadManifest);
all('[data-mode]').forEach((button) => button.addEventListener('click', () => {
  state.mode = button.dataset.mode;
  for (const viewer of Object.values(viewers)) viewer.setMode(state.mode);
  all('[data-mode]').forEach((entry) => entry.setAttribute('aria-pressed', String(entry === button)));
}));
all('[data-view]').forEach((button) => button.addEventListener('click', () => setView(button.dataset.view)));
$('mouth-focus').addEventListener('click', () => focus('mouth'));
$('reset-view').addEventListener('click', () => focus('face'));
$('exposure').addEventListener('input', (event) => {
  const value = Number(event.target.value);
  $('exposure-value').textContent = value.toFixed(2);
  for (const viewer of Object.values(viewers)) viewer.setExposure(value);
});
window.addEventListener('pagehide', () => { for (const viewer of Object.values(viewers)) viewer.dispose(); }, { once: true });
controls();
await loadManifest();
