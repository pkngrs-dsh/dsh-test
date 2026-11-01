import * as THREE from 'three';
import { FaceViewer } from '/viewer-common/scene.js';

const $ = (id) => document.getElementById(id);
const all = (selector) => [...document.querySelectorAll(selector)];
const branchNames = { source: '原始 UFF · 参考', baseline: '当前线上 UV 搬运', before: '处理前 · 无模板形状先验', mouthOnly: '后处理 · 仅嘴部', frontLandmarks: '分区后处理 · 嘴 → 左眼 → 右眼', jointFrontLandmarks: '同时拟合 / 整体回退', rawFrontLandmarks: '完整拟合 · 未回退（诊断）' };
const caseNames = { maleExactRegression: '歪嘴回归 · 男性模板', orbitControl4: '环绕基线 · 女性模板', highPitch4: '明显抬头 · 扭头组', cyanWall4: '浅蓝背景 · 低清扭头组', outdoorSideLight4: '室外方向光 · 低清扭头组', backlitCap4: '逆光帽檐 · 挑战组' };
const state = { manifest: null, preview: null, appearance: null, appearanceNotice: '', appearanceErrors: {}, textureBlobs: new Map(), selected: null, busy: false, generation: 0, mode: 'textured', frame: null, focus: 'face', syncing: false, imageName: 'texturedComparison.png' };
const viewers = {};

function status(text, error = false) { $('status').textContent = text; $('status').classList.toggle('error', error); }
function controls() {
  for (const id of ['case-select', 'left-branch', 'right-branch', 'mouth-focus', 'reset-view', 'exposure']) $(id).disabled = state.busy || !state.preview;
  all('[data-mode], [data-view]').forEach((button) => { button.disabled = state.busy || !state.preview; });
  $('reload').disabled = state.busy;
  $('exposure').disabled = state.busy || !state.preview || state.mode === 'baseColor';
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

async function fetchAppearance(id, preview) {
  state.appearanceNotice = '';
  state.appearanceErrors = {};
  const response = await fetch(apiPath(`/data/${id}/appearance.json`), { cache: 'no-store', credentials: 'same-origin' });
  if (response.status === 404) {
    state.appearanceNotice = '此组尚未生成贴图资源，当前显示灰模。';
    return null;
  }
  if (!response.ok) throw new Error(`无法读取贴图说明（HTTP ${response.status}）。`);
  const appearance = await response.json();
  if (appearance.schemaVersion !== 1 || !appearance.branches || typeof appearance.branches !== 'object') throw new Error('贴图说明格式无效。');
  const branches = {};
  for (const branch of preview.branches) {
    try {
      const entry = appearance.branches[branch.id];
      if (!entry) throw new Error('此分支尚无贴图');
      const { uvs, uvFaces } = entry.mesh || {};
      if (!Array.isArray(uvs) || !uvs.length || !uvs.every((uv) => Array.isArray(uv) && uv.length === 2 && uv.every(Number.isFinite))
          || !Array.isArray(uvFaces) || uvFaces.length !== branch.faces.length
          || !uvFaces.every((face) => Array.isArray(face) && face.length === 3 && face.every((index) => Number.isInteger(index) && index >= 0 && index < uvs.length))) {
        throw new Error('UV 或 UV 三角面与网格不匹配');
      }
      const texture = entry.texture;
      if (!texture || typeof texture.sha256 !== 'string' || !/^[0-9a-f]{64}$/i.test(texture.sha256) || !Number.isSafeInteger(texture.sizeBytes) || texture.sizeBytes <= 0 || texture.colorSpace !== 'sRGB') {
        throw new Error('贴图校验信息无效');
      }
      apiPath(texture.url);
      branches[branch.id] = entry;
    } catch (error) {
      state.appearanceErrors[branch.id] = error.message;
    }
  }
  return { ...appearance, branches };
}

function textureResolver(descriptor) {
  return async (url) => {
    const resolved = apiPath(url);
    if (resolved !== apiPath(descriptor.url)) throw new Error('贴图地址与说明不一致。');
    const key = `${resolved}|${descriptor.sha256.toLowerCase()}|${descriptor.sizeBytes}`;
    if (!state.textureBlobs.has(key)) {
      const pending = (async () => {
        const response = await fetch(resolved, { cache: 'no-store', credentials: 'same-origin' });
        if (!response.ok) throw new Error(`贴图读取失败（HTTP ${response.status}）。`);
        const bytes = await response.arrayBuffer();
        if (bytes.byteLength !== descriptor.sizeBytes) throw new Error('贴图文件大小校验失败。');
        if (!globalThis.crypto?.subtle) throw new Error('当前浏览器无法校验贴图 SHA-256。');
        const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (byte) => byte.toString(16).padStart(2, '0')).join('');
        if (digest !== descriptor.sha256.toLowerCase()) throw new Error('贴图 SHA-256 校验失败。');
        return new Blob([bytes], { type: response.headers.get('content-type') || 'image/png' });
      })();
      state.textureBlobs.set(key, pending);
    }
    return state.textureBlobs.get(key);
  };
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

function branchPayload(branch, appearance = null) {
  const { vertices, faces } = branch;
  if (!Array.isArray(vertices) || !vertices.length || !vertices.every((v) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite))
      || !Array.isArray(faces) || !faces.length || !faces.every((f) => Array.isArray(f) && f.length === 3 && f.every((i) => Number.isInteger(i) && i >= 0 && i < vertices.length))) {
    throw new Error('实验网格的顶点或三角面无效。');
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices.flat(), 3));
  geometry.setIndex(faces.flat());
  geometry.computeVertexNormals();
  const sourceNormals = Array.from(geometry.getAttribute('normal').array);
  geometry.dispose();
  let mesh;
  if (appearance) {
    const { uvs: sourceUvs, uvFaces } = appearance.mesh;
    const positions = [], normals = [], uvs = [], indices = [], sourceVertexIds = [], pairs = new Map();
    faces.forEach((face, faceId) => face.forEach((vertexId, corner) => {
      const uvId = uvFaces[faceId][corner], key = `${vertexId}:${uvId}`;
      let index = pairs.get(key);
      if (index === undefined) {
        index = sourceVertexIds.length;
        pairs.set(key, index);
        sourceVertexIds.push(vertexId);
        positions.push(...vertices[vertexId]);
        normals.push(...sourceNormals.slice(vertexId * 3, vertexId * 3 + 3));
        // The asset compiler supplies V-down UVs; FaceViewer keeps flipY=false.
        uvs.push(...sourceUvs[uvId]);
      }
      indices.push(index);
    }));
    mesh = { positions, normals, uvs, indices, sourceVertexIds };
  } else {
    // FaceViewer requires a UV attribute even for a material with no texture.
    // These placeholders are only used by the explicitly untextured fallback.
    mesh = { positions: vertices.flat(), normals: sourceNormals, indices: faces.flat(), uvs: Array(vertices.length * 2).fill(0), sourceVertexIds: vertices.map((_, index) => index) };
  }
  mesh.groups = [{ start: 0, count: mesh.indices.length, materialIndex: 0 }];
  return { templateId: branch.id, coordinates: { lengthUnit: 'centimeter', frame: 'shared baseline template' },
    mesh, materials: [{ index: 0, name: branch.id, alphaMode: 'OPAQUE', baseColorFactor: appearance ? [1, 1, 1, 1] : [0.6, 0.7, 0.63, 1], roughnessFactor: 0.85, metallicFactor: 0,
      ...(appearance ? { texture: appearance.texture } : {}) }] };
}

function applyRenderMode(viewer) {
  const asset = viewer.asset;
  const textured = Boolean(asset?.modes.textured.some((material) => material.map));
  const mode = ['textured', 'baseColor'].includes(state.mode) && !textured ? 'clay' : state.mode;
  // Keep the shared viewer's declared mode valid so its next load is also valid.
  viewer.setMode(mode === 'baseColor' ? 'textured' : mode);
  if (mode === 'baseColor' && asset) {
    if (!asset.modes.baseColor) {
      asset.modes.baseColor = asset.modes.textured.map((material) => new THREE.MeshBasicMaterial({
        name: `${material.name}:baseColor`, color: 0xffffff, map: material.map, opacity: 1,
        transparent: false, side: THREE.FrontSide, toneMapped: false,
      }));
      asset.materials.push(...asset.modes.baseColor);
    }
    asset.mesh.material = asset.modes.baseColor;
  }
}

function showAppearanceInfo(selected, notices) {
  const entries = ['left', 'right'].map((side) => state.appearance?.branches[selected[side].id]);
  const sameRows = (left, right) => left.length === right.length && left.every((row, index) => row.length === right[index].length && row.every((value, corner) => value === right[index][corner]));
  const sameAppearance = entries.every(Boolean) && entries[0].texture.sha256.toLowerCase() === entries[1].texture.sha256.toLowerCase()
    && sameRows(entries[0].mesh.uvs, entries[1].mesh.uvs) && sameRows(entries[0].mesh.uvFaces, entries[1].mesh.uvFaces);
  const hasSource = selected.left.id === 'source' || selected.right.id === 'source';
  $('appearance-status').textContent = notices.length ? notices.join('；')
    : hasSource ? '当前包含原始 UFF 参考：其 UV 与贴图不同于客户端模板，仅供参考。'
      : sameAppearance ? '两侧使用同一张原融合贴图和原有 UV；只比较几何变化。'
        : '当前两侧贴图或 UV 不一致；此组合不能单独比较几何变化。';
  $('appearance-status').classList.toggle('fallback', notices.length > 0);
  for (const side of ['left', 'right']) {
    const entry = state.appearance?.branches[selected[side].id];
    $(`${side}-texture-tag`).textContent = notices.some((notice) => notice.startsWith(side === 'left' ? '左侧' : '右侧')) || !entry
      ? '无可用贴图 · 灰模' : entry.provenanceLabel || '已载入贴图';
  }
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
function numberOf(value) {
  if (Number.isFinite(value)) return value;
  if (value && Number.isFinite(value.meanErrorPixels)) return value.meanErrorPixels;
  return null;
}
function errorValue(value) {
  const number = numberOf(value);
  return number === null ? '—' : number.toFixed(2);
}
function pairCount(value) {
  if (Number.isFinite(value)) return value;
  if (Array.isArray(value)) return value.length;
  if (value && Number.isFinite(value.positiveLengthOrAreaPairs)) return value.positiveLengthOrAreaPairs;
  return null;
}
function renderMetrics(side, branch) {
  const container = $(`${side}-metrics`), metrics = branch.metrics || {};
  let errors = metrics.frontErrors || {};
  // The candidate's "before" observations refer to this exact starting branch,
  // never to source UFF or the online UV-transfer mesh.
  if (branch.id === 'before' && numberOf(errors.mouthBefore) === null && numberOf(errors.eyesBefore) === null) {
    errors = state.preview?.branches.find((item) => item.id === 'frontLandmarks')?.metrics?.frontErrors || errors;
  }
  container.replaceChildren();
  $(`${side}-tag`).textContent = branchNames[branch.id] || branch.label || branch.id;
  const before = ['source', 'baseline', 'before'].includes(branch.id);
  const mouth = before ? errors.mouthBefore ?? errors.mouthAfter : errors.mouthAfter;
  const eyes = before ? errors.eyesBefore ?? errors.eyesAfter : errors.eyesAfter;
  if (numberOf(mouth) !== null || numberOf(eyes) !== null) {
    metric(container, '正面嘴部误差 / px', errorValue(mouth));
    metric(container, '正面眼部误差 / px', errorValue(eyes));
    if (!before && numberOf(errors.mouthBefore) !== null) metric(container, '嘴部处理前 / px', errorValue(errors.mouthBefore));
    if (!before && numberOf(errors.eyesBefore) !== null) metric(container, '眼部处理前 / px', errorValue(errors.eyesBefore));
  }
  const collision = metrics.collision || {};
  const count = pairCount(before ? collision.before ?? collision.after : collision.after);
  const added = pairCount(collision.newPairs);
  if (count !== null) metric(container, '非邻接面穿插对', String(count), count > 0);
  if (!before && added !== null) metric(container, '相对处理前新增对', String(added), added > 0);
  const percent = (ratio) => `${new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 }).format(ratio * 100)}%`;
  if (!before && collision.stageScales && typeof collision.stageScales === 'object') {
    for (const [key, label] of Object.entries({ mouth: '嘴部保留位移', imageLeftEye: '画面左眼保留位移', imageRightEye: '画面右眼保留位移' })) {
      const ratio = collision.stageScales[key];
      if (Number.isFinite(ratio)) metric(container, label, percent(ratio), ratio === 0);
    }
  } else if (!before && Number.isFinite(collision.scale)) {
    metric(container, '整体保留位移', percent(collision.scale), collision.scale === 0);
  }
  if (Number.isFinite(metrics.elapsedSeconds)) metric(container, '后处理 / 秒', metrics.elapsedSeconds.toFixed(3));
  if (!container.children.length) {
    metric(container, '顶点', branch.vertices.length.toLocaleString('zh-CN'));
    metric(container, '三角面', branch.faces.length.toLocaleString('zh-CN'));
  }
  if (metrics.fitStatus) metric(container, '拟合状态', ({ acceptedLocalCandidate: '已应用局部校正', unchanged: '保留处理前结果', diagnosticOnly: '仅供诊断' })[metrics.fitStatus] || '本地候选');
  const note = document.createElement('span');
  note.className = 'metric-status';
  note.textContent = branch.id === 'source' ? '原始 UFF 为输入参考，拓扑不同，穿插对数不能与客户端直接比较。'
    : branch.id === 'baseline' ? '当前线上方法作为对照；此分支未记录同口径的正面关键点误差。后处理尚未部署。'
    : branch.id === 'before' ? '无模板形状先验，固定接口；后处理以此分支为起点。'
    : branch.id === 'rawFrontLandmarks' ? '完整位移诊断分支，未做新增穿插回退；不可视为可发布结果。'
    : branch.id === 'frontLandmarks' ? '按嘴部、画面左眼、画面右眼依次调整，各部位独立回退；已有自交未消除，也未保证其严重程度不增加。'
    : branch.id === 'jointFrontLandmarks' ? '嘴眼同时拟合，再统一缩小整张脸的位移；用于与分区后处理对照。已有自交及其严重程度仍需检查。'
    : '正面误差针对自动观测点；回退仅限制新增的最终相交面片对，未消除已有自交，也未保证其严重程度。';
  if (branch.id === 'rawFrontLandmarks') note.classList.add('failed');
  container.append(note);
}
function showEvidence(name = state.imageName) {
  if (!state.selected) return;
  state.imageName = name;
  const labels = { 'texturedComparison.png': '原照片、原始 UFF、处理前与处理后的原相机贴图基色对照，不叠加场景光照。客户端两分支共用原融合贴图，尚未重投影。点击查看原大小。',
    'texturedLitComparison.png': '原照片、原始 UFF、处理前与处理后的原相机贴图光照对照，各三维分支使用同一光照。客户端两分支共用原融合贴图，尚未重投影。点击查看原大小。',
    'frontOverlay.png': '照片自动检测点与网格投影对照；图中颜色含义以图内标注为准。点击查看原大小。',
    'mouth.png': '同一镜头与光照下的嘴部几何对照。点击查看原大小。',
    'face.png': '同一镜头与光照下的全脸几何对照。点击查看原大小。',
    'side.png': '侧面几何对照；固定正面深度不代表侧面轮廓完全不变。点击查看原大小。' };
  if (!Object.hasOwn(labels, name)) return;
  const image = $('evidence-image'), url = apiPath(`/data/${state.selected}/${name}`);
  image.hidden = true;
  $('image-status').hidden = false;
  $('image-status').textContent = '正在读取图片…';
  $('image-description').textContent = labels[name];
  $('image-full').href = url;
  image.alt = `${caseNames[state.selected] || state.selected} · ${labels[name]}`;
  image.onload = () => { if (image.src !== url) return; image.hidden = false; $('image-status').hidden = true; };
  image.onerror = () => { if (image.src !== url) return; image.hidden = true; $('image-status').hidden = false; $('image-status').textContent = '此对照图尚未生成。三维预览仍可使用。'; };
  image.src = url;
  all('[data-image]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.image === name)));
}

async function showBranches(keepView = true) {
  if (!state.preview || state.busy || !viewers.left || !viewers.right) return;
  const selected = Object.fromEntries(['left', 'right'].map((side) => [side, state.preview.branches.find((branch) => branch.id === $(`${side}-branch`).value)]));
  if (!selected.left || !selected.right) return;
  const pose = keepView ? snapshot(viewers.left) : null;
  state.busy = true;
  controls();
  status('正在载入两侧几何与贴图…');
  try {
    const notices = state.appearanceNotice ? [state.appearanceNotice] : [];
    await Promise.all(['left', 'right'].map(async (side) => {
      const branch = selected[side], appearance = state.appearance?.branches[branch.id];
      const label = side === 'left' ? '左侧' : '右侧';
      try {
        if (!appearance) {
          if (!state.appearanceNotice) notices.push(`${label}：${state.appearanceErrors[branch.id] || '没有可用贴图'}，显示灰模`);
          await viewers[side].load(branchPayload(branch), async () => { throw new Error('未声明贴图资源。'); });
        } else {
          await viewers[side].load(branchPayload(branch, appearance), textureResolver(appearance.texture));
        }
      } catch (error) {
        notices.push(`${label}：${error.message} 已回退灰模`);
        await viewers[side].load(branchPayload(branch), async () => { throw new Error('未声明贴图资源。'); });
      }
    }));
    applyFrame();
    if (pose) {
      for (const viewer of Object.values(viewers)) restore(viewer, pose);
    } else {
      viewers.left.setView('front');
      restore(viewers.right, snapshot(viewers.left));
      state.focus = 'face';
    }
    for (const side of ['left', 'right']) { applyRenderMode(viewers[side]); renderMetrics(side, selected[side]); }
    showAppearanceInfo(selected, notices);
    status('两侧镜头同步 · 几何校正后尚未重投影纹理，原融合贴图可能仍有错位 · 尚未替换远程服务。');
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
  status('正在读取此组的后处理对照…');
  try {
    const preview = await fetchJSON(apiPath(entry.previewUrl || `/data/${id}/preview.json`));
    if (generation !== state.generation) return;
    if (preview.caseId !== id || !Array.isArray(preview.branches) || !preview.branches.length
        || !Array.isArray(preview.mouthCenter) || preview.mouthCenter.length !== 3 || !preview.mouthCenter.every(Number.isFinite)) throw new Error('实验预览格式无效。');
    preview.branches.forEach((branch) => branchPayload(branch));
    state.textureBlobs.clear();
    try {
      state.appearance = await fetchAppearance(id, preview);
    } catch (error) {
      state.appearance = null;
      state.appearanceNotice = `${error.message} 当前显示灰模。`;
    }
    state.preview = preview;
    state.selected = id;
    state.frame = commonFrame(preview);
    $('case-select').value = id;
    for (const side of ['left', 'right']) {
      const select = $(`${side}-branch`);
      const wanted = preferences[side] || select.value || (side === 'left' ? 'before' : 'frontLandmarks');
      select.replaceChildren();
      for (const branch of preview.branches) select.add(new Option(branchNames[branch.id] || branch.label || branch.id, branch.id));
      select.value = preview.branches.some((branch) => branch.id === wanted) ? wanted : preview.branches[0].id;
    }
    $('case-title').textContent = caseNames[id] || entry.label || id;
    const capture = entry.captureMode === 'userConfirmedCameraOrbit' ? '相机环绕（用户已确认）' : '该批次主要扭头，未逐目录标定';
    const gender = entry.gender === 'male' ? '男性实验模板' : entry.gender === 'female' ? '女性实验模板' : '模板选择见报告';
    $('case-description').textContent = `${gender} · ${capture}。照片第 1 帧为预选正面参考，保留其原有俯仰与侧倾。`;
    for (const [element, filename] of Object.entries({ 'overlay-image': 'frontOverlay.png', 'mouth-image': 'mouth.png', 'side-image': 'side.png', 'report-link': 'report.json', 'mesh-download': 'meshes.npz' })) {
      $(element).href = apiPath(`/data/${id}/${filename}`);
    }
    showEvidence();
  } catch (error) { status(error.message, true); state.preview = null; }
  finally { state.busy = false; controls(); }
  if (state.preview && generation === state.generation) await showBranches(false);
}

async function loadManifest() {
  if (state.busy) return;
  state.busy = true;
  controls();
  try {
    const manifest = await fetchJSON('/api/experiment');
    if (!Array.isArray(manifest.cases) || !manifest.cases.length) throw new Error('尚未生成可预览的实验组，请稍后刷新。');
    state.manifest = manifest;
    $('case-select').replaceChildren();
    manifest.cases.forEach((entry) => $('case-select').add(new Option(caseNames[entry.id] || entry.label || entry.id, entry.id)));
    $('objective').textContent = `实验目标：${manifest.objective || '通过固定相机中的正面关键点，局部修正直接配准后的客户端几何。'}`;
    $('solver').textContent = `求解方式：${manifest.solver || '局部位移平滑，保持正面相机深度；具体约束、权重与回退结果见每组报告。'}`;
  } catch (error) { status(error.message, true); }
  finally { state.busy = false; controls(); }
  if (state.manifest?.cases.length) {
    const query = new URL(location.href).searchParams;
    const wanted = state.selected || query.get('case');
    await chooseCase(state.manifest.cases.some((entry) => entry.id === wanted) ? wanted : state.manifest.cases[0].id,
      { left: query.get('left') || 'before', right: query.get('right') || 'frontLandmarks' });
  }
}

$('case-select').addEventListener('change', (event) => chooseCase(event.target.value));
$('left-branch').addEventListener('change', () => showBranches());
$('right-branch').addEventListener('change', () => showBranches());
$('reload').addEventListener('click', loadManifest);
all('[data-image]').forEach((button) => button.addEventListener('click', () => showEvidence(button.dataset.image)));
all('[data-mode]').forEach((button) => button.addEventListener('click', () => {
  state.mode = button.dataset.mode;
  for (const viewer of Object.values(viewers)) applyRenderMode(viewer);
  all('[data-mode]').forEach((entry) => entry.setAttribute('aria-pressed', String(entry === button)));
  controls();
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
