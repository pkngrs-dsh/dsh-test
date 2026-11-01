import { BrowserClient, ApiError } from './api.js';
import { FaceViewer } from './scene.js';

const $ = (id) => document.getElementById(id);
const all = (selector) => [...document.querySelectorAll(selector)];
const JOB_ID = /^[0-9a-f]{32}$/;
const STORAGE_KEY = 'faceAppearance.viewer.lastJobId.v1';
const api = new BrowserClient();
const state = { files: [], capabilities: null, job: null, submitting: false,
  loading: false, previewLoaded: false, downloading: false, generation: 0,
  controller: null, downloadController: null, timer: null, pollFailures: 0, connecting: false };
let viewer;
let viewerError;
try {
  viewer = new FaceViewer($('viewport'), { onError(error) {
    viewerError = error;
    state.previewLoaded = false;
    overlay('三维预览暂不可用', error.message);
    updateControls();
  } });
} catch (error) {
  viewerError = error;
  overlay('浏览器未能启用 WebGL', '请使用支持 WebGL 2 的浏览器并开启硬件加速。你仍可提交任务和下载资产。');
}

function notice(message) {
  $('notice-text').textContent = message;
  $('notice').hidden = false;
}

function errorMessage(error) {
  const messages = {
    unauthorized: '服务需要有效的访问令牌，请在右上角连接设置中填写。',
    queue_full: '服务队列已满，请稍后再次提交。',
    not_found: '未找到此任务，请核对任务 ID 和当前服务的作业目录。',
    template_mismatch: '服务模板已变更，请重新连接后提交。',
    service_restarted: '服务在任务完成前重启，此任务已结束。请重新提交照片。',
    invalid_client_asset: '资产包未通过完整性检查，请检查服务日志。',
    reconstruction_failed: '重建失败，请检查照片，并使用任务 ID 查看服务日志。',
    client_service_unavailable: '服务尚未准备好客户端装配，请稍后重新连接。',
    service_unavailable: '服务暂时不可用，请稍后重新连接。',
    result_not_ready: '资产仍在生成，请继续读取任务。',
  };
  if (error?.name === 'TimeoutError') return '请求超时，请稍后继续读取任务。';
  if (error instanceof TypeError && /fetch|network/i.test(error.message)) return '无法连接服务，请检查服务进程和网络。';
  return messages[error?.code] || error?.message || '请求失败，请稍后重试。';
}

function showError(error) {
  notice(errorMessage(error));
  if (error?.status === 401) {
    $('connection').textContent = '需要令牌';
    $('connection').dataset.state = 'error';
    state.capabilities = null;
  }
  updateControls();
}

function overlay(title, message, loading = false) {
  $('viewport-overlay').hidden = false;
  $('viewport-overlay').classList.toggle('loading', loading);
  $('overlay-title').textContent = title;
  $('overlay-message').textContent = message;
}

function limits() {
  return state.capabilities?.limits || { minImages: 4, maxImages: 16,
    maxImageBytes: 25 * 1024 ** 2, maxTotalBytes: 200 * 1024 ** 2 };
}

function mib(bytes) { return `${(bytes / 1024 ** 2).toFixed(1)} MiB`; }
function activeJob() { return ['queued', 'running'].includes(state.job?.status); }

function uploadProblem() {
  const rules = limits();
  if (state.files.length > rules.maxImages) return `当前服务最多接受 ${rules.maxImages} 张照片，请移除多余照片。`;
  if (state.files.some((entry) => entry.file.size > rules.maxImageBytes)) return '部分照片超过当前服务的单张大小限制。';
  if (state.files.reduce((sum, entry) => sum + entry.file.size, 0) > rules.maxTotalBytes) return '照片总大小超过当前服务限制。';
  return null;
}

function updateControls() {
  const min = limits().minImages;
  const active = state.submitting || activeJob();
  $('submit').disabled = active || state.loading || state.connecting || !state.capabilities?.ready
    || state.files.length < min || !state.capabilities?.template || Boolean(uploadProblem());
  $('submit').querySelector('span').textContent = state.submitting ? '正在上传照片…' : activeJob() ? '任务处理中…' : '开始重建';
  $('submit-hint').textContent = active ? '任务已提交后会自动轮询，无需重复上传。'
    : !state.capabilities?.ready ? '连接服务后，选择照片开始重建。'
    : uploadProblem() || (state.files.length < min ? `还需选择 ${min - state.files.length} 张照片。`
    : `已选择 ${state.files.length} 张照片，可以开始重建。`);
  $('photos').disabled = state.submitting;
  $('dropzone').dataset.disabled = String(state.submitting);
  $('clear-files').disabled = state.submitting;
  all('.thumbnail button').forEach((button) => { button.disabled = state.submitting; });
  $('settings').disabled = state.submitting;
  $('resume').disabled = state.submitting || state.loading || state.connecting;
  $('retry').disabled = state.submitting || state.loading || state.connecting;
  $('download').disabled = state.downloading || state.job?.status !== 'succeeded' || !state.job.download;
  $('download').querySelector('span').textContent = state.downloading ? '下载并校验中…' : '下载资产 ZIP';
  all('[data-mode], [data-view], #reset-view, #auto-rotate, #exposure, #screenshot').forEach((button) => {
    button.disabled = !state.previewLoaded || Boolean(viewerError);
  });
}

function renderFiles() {
  $('thumbnails').replaceChildren();
  state.files.forEach((entry, index) => {
    const item = document.createElement('div');
    item.className = 'thumbnail';
    const img = document.createElement('img');
    img.src = entry.url;
    img.alt = entry.file.name;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.textContent = '×';
    remove.setAttribute('aria-label', `移除 ${entry.file.name}`);
    remove.addEventListener('click', () => {
      if (state.submitting) return;
      URL.revokeObjectURL(entry.url);
      state.files.splice(index, 1);
      renderFiles();
    });
    item.append(img, remove);
    $('thumbnails').append(item);
  });
  $('file-count').textContent = `${state.files.length} / ${limits().maxImages}`;
  $('upload-hint').textContent = `${limits().minImages}–${limits().maxImages} 张 · 单张 ≤ ${Math.floor(limits().maxImageBytes / 1024 ** 2)} MiB`;
  $('files-toolbar').hidden = !state.files.length;
  $('file-size').textContent = `共 ${mib(state.files.reduce((sum, entry) => sum + entry.file.size, 0))}`;
  updateControls();
}

function addFiles(files) {
  if (state.submitting) return;
  const rules = limits();
  const skipped = [];
  let total = state.files.reduce((sum, entry) => sum + entry.file.size, 0);
  for (const file of files) {
    if (state.files.some((entry) => entry.file.name === file.name && entry.file.size === file.size
        && entry.file.lastModified === file.lastModified)) continue;
    if (!/\.(jpe?g|png|webp|bmp|tiff?)$/i.test(file.name)) { skipped.push(`${file.name}：不支持的图片格式`); continue; }
    if (!file.size || file.size > rules.maxImageBytes) { skipped.push(`${file.name}：图片大小不符合要求`); continue; }
    if (state.files.length >= rules.maxImages) { skipped.push(`最多选择 ${rules.maxImages} 张照片`); break; }
    if (total + file.size > rules.maxTotalBytes) { skipped.push(`照片总大小不能超过 ${mib(rules.maxTotalBytes)}`); continue; }
    state.files.push({ file, url: URL.createObjectURL(file) });
    total += file.size;
  }
  renderFiles();
  if (skipped.length) notice(skipped.slice(0, 3).join('；'));
}

function remember(jobId) {
  try { localStorage.setItem(STORAGE_KEY, jobId); } catch { /* Storage can be disabled. */ }
  $('resume-id').value = jobId;
  const url = new URL(location.href);
  url.searchParams.set('job', jobId);
  history.replaceState(null, '', url);
}

function taskStatus(status, message) {
  const labels = { idle: '未开始', uploading: '上传中', queued: '排队中', running: '重建中',
    succeeded: '已完成', failed: '失败', loading: '读取中', interrupted: '连接中断' };
  $('task-pill').textContent = labels[status] || status;
  $('task-pill').dataset.state = status;
  $('task-message').textContent = message;
  const progress = { uploading: 0, queued: 1, running: 1, succeeded: 3, failed: 1 }[status] ?? -1;
  all('.task-progress i').forEach((step, index) => {
    step.classList.toggle('done', index < progress);
    step.classList.toggle('active', index === progress && ['uploading', 'queued', 'running'].includes(status));
  });
}

function resetPreview() {
  viewer?.clear();
  state.previewLoaded = false;
  $('preview-label').textContent = '等待资产';
  $('asset-name').textContent = '尚未生成资产';
  $('asset-description').textContent = '任务完成后可预览，并下载完整资产包。';
  for (const id of ['source-vertices', 'render-vertices', 'face-count', 'material-count']) $(id).textContent = '—';
  $('viewport-state').textContent = '本地 Three.js 预览';
  $('auto-rotate').checked = false;
  viewer?.setAutoRotate(false);
}

function newOperation() {
  clearTimeout(state.timer);
  state.controller?.abort();
  state.downloadController?.abort();
  state.downloadController = null;
  state.downloading = false;
  state.controller = new AbortController();
  state.generation += 1;
  state.pollFailures = 0;
  $('retry').hidden = true;
  return { generation: state.generation, signal: state.controller.signal };
}

function current(operation) {
  return operation.generation === state.generation && !operation.signal.aborted;
}

async function loadPreview(operation) {
  if (!viewer || viewerError) {
    overlay('浏览器未能启用三维预览', '资产已生成，可下载 ZIP。预览需要支持 WebGL 2 的浏览器。');
    return;
  }
  state.loading = true;
  overlay('正在载入三维资产', '读取网格、校验贴图并准备材质…', true);
  updateControls();
  try {
    const job = state.job;
    const preview = await api.getPreview(job.jobId, { signal: operation.signal });
    if (!current(operation)) return;
    const hashes = Object.entries(job.asset?.templateHashes || {});
    if (preview.templateId !== job.asset?.templateId || !hashes.length
        || hashes.length !== Object.keys(preview.templateHashes || {}).length
        || hashes.some(([key, hash]) => preview.templateHashes[key] !== hash)) {
      throw new Error('预览模板身份与当前任务不匹配。');
    }
    const textures = new Map(preview.materials.filter((material) => material.texture)
      .map((material) => [material.texture.url, material.texture]));
    await viewer.load(preview, (url) => api.getTexture(textures.get(url), { signal: operation.signal }));
    if (!current(operation)) return;
    state.previewLoaded = true;
    $('viewport-overlay').hidden = true;
    $('preview-label').textContent = '资产已载入';
    $('viewport-state').textContent = '纹理已校验 · 工作室光照';
    $('asset-name').textContent = preview.templateId;
    $('asset-description').textContent = `米制坐标 · ${mib(job.download.sizeBytes)} 资产包`;
    const fields = { 'source-vertices': 'sourceVertices', 'render-vertices': 'renderVertices',
      'face-count': 'faces', 'material-count': 'materials' };
    for (const [id, key] of Object.entries(fields)) $(id).textContent = Number(preview.counts[key]).toLocaleString('zh-CN');
    if (preview.warnings?.includes('preview_contains_zero_length_normals')) notice('资产存在零长度法线，部分预览光照可能异常。');
  } catch (error) {
    if (!current(operation) || error.name === 'AbortError') return;
    overlay('资产预览未能载入', errorMessage(error));
    $('retry').hidden = false;
    showError(error);
  } finally {
    if (current(operation)) { state.loading = false; updateControls(); }
  }
}

async function acceptJob(job, operation) {
  if (!current(operation)) return;
  state.job = job;
  remember(job.jobId);
  $('job-identity').hidden = false;
  $('job-id').textContent = job.jobId;
  $('retry').hidden = true;
  if (job.status === 'queued') {
    taskStatus('queued', '任务已接收，正在等待 GPU。可以关闭页面，之后通过任务 ID 恢复。');
    overlay('任务已进入队列', '服务将按顺序处理，页面会自动更新。', true);
  } else if (job.status === 'running') {
    taskStatus('running', '正在重建几何、融合贴图并装配资产。完成后会自动载入预览。');
    overlay('正在重建三维头部', '几何重建 → 纹理融合 → 资产装配', true);
  } else if (job.status === 'failed') {
    taskStatus('failed', errorMessage(job.error));
    overlay('本次重建未完成', errorMessage(job.error));
  } else {
    taskStatus('succeeded', '资产已生成，可检查纹理和几何，并下载完整 ZIP。');
    $('asset-name').textContent = job.asset?.templateId || '客户端资产';
    $('asset-description').textContent = job.download ? `${mib(job.download.sizeBytes)} 资产包` : '';
  }
  updateControls();
  if (activeJob()) {
    const seconds = Math.max(1, Math.min(Number(job.pollAfterSeconds) || 2, 30));
    state.timer = setTimeout(() => poll(job.jobId, operation), seconds * 1000);
  } else if (job.status === 'succeeded') await loadPreview(operation);
}

async function poll(jobId, operation) {
  try {
    const job = await api.getJob(jobId, { signal: operation.signal });
    if (!current(operation)) return;
    state.pollFailures = 0;
    await acceptJob(job, operation);
  } catch (error) {
    if (!current(operation) || error.name === 'AbortError') return;
    const transient = !(error instanceof ApiError) || error.retryable;
    state.pollFailures += 1;
    if (transient && state.pollFailures <= 5) {
      const delay = Math.max(2, Math.min(error.retryAfter || 2 ** state.pollFailures, 30));
      taskStatus('interrupted', `连接暂时中断，${delay} 秒后继续查询同一任务。`);
      state.timer = setTimeout(() => poll(jobId, operation), delay * 1000);
    } else {
      taskStatus('interrupted', `${errorMessage(error)} 任务 ID 已保留，可继续读取。`);
      $('retry').hidden = false;
      showError(error);
    }
  }
}

async function resume(jobId) {
  if (state.submitting) return;
  if (!JOB_ID.test(jobId)) { notice('请输入 32 位小写十六进制任务 ID。'); return; }
  const operation = newOperation();
  state.job = null;
  state.loading = true;
  resetPreview();
  $('job-identity').hidden = true;
  taskStatus('loading', '正在读取任务状态…');
  overlay('正在读取任务', '加载已保存的任务状态和资产。', true);
  updateControls();
  try {
    const job = await api.getJob(jobId, { signal: operation.signal });
    if (!current(operation)) return;
    state.loading = false;
    await acceptJob(job, operation);
  } catch (error) {
    if (!current(operation) || error.name === 'AbortError') return;
    taskStatus('interrupted', errorMessage(error));
    overlay('暂时无法读取任务', errorMessage(error));
    $('retry').hidden = false;
    showError(error);
  } finally {
    if (current(operation)) { state.loading = false; updateControls(); }
  }
}

async function connect() {
  state.connecting = true;
  $('connect').disabled = true;
  $('connection-error').textContent = '';
  $('connection').textContent = '正在连接';
  $('connection').dataset.state = 'pending';
  updateControls();
  try {
    const caps = await api.getCapabilities();
    if (!caps.limits || !Number.isInteger(caps.limits.minImages)
        || !Number.isInteger(caps.limits.maxImages) || caps.limits.minImages < 1
        || caps.limits.maxImages < caps.limits.minImages || !Number.isFinite(caps.limits.maxImageBytes)
        || !Number.isFinite(caps.limits.maxTotalBytes)) throw new Error('服务未返回有效的上传限制。');
    state.capabilities = caps;
    $('connection').textContent = caps.ready ? '服务已连接' : '服务未就绪';
    $('connection').dataset.state = caps.ready ? 'ready' : 'error';
    if (caps.ready) $('connection-dialog').close();
    else $('connection-error').textContent = '服务尚未准备好客户端装配，请检查配置或稍后重连。';
    renderFiles();
    return true;
  } catch (error) {
    state.capabilities = null;
    $('connection').textContent = error.status === 401 ? '需要令牌' : '服务未连接';
    $('connection').dataset.state = 'error';
    $('connection-error').textContent = errorMessage(error);
    return false;
  } finally {
    state.connecting = false;
    $('connect').disabled = false;
    updateControls();
  }
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

$('photos').addEventListener('change', (event) => { addFiles(event.target.files); event.target.value = ''; });
for (const name of ['dragenter', 'dragover']) $('dropzone').addEventListener(name, (event) => {
  event.preventDefault(); if (!state.submitting) $('dropzone').classList.add('dragover');
});
for (const name of ['dragleave', 'drop']) $('dropzone').addEventListener(name, (event) => {
  event.preventDefault(); $('dropzone').classList.remove('dragover');
  if (name === 'drop') addFiles(event.dataTransfer.files);
});
$('clear-files').addEventListener('click', () => {
  if (state.submitting) return;
  state.files.forEach((entry) => URL.revokeObjectURL(entry.url)); state.files = []; renderFiles();
});
$('submit').addEventListener('click', async () => {
  if ($('submit').disabled) return;
  const operation = newOperation();
  state.submitting = true;
  state.job = null;
  resetPreview();
  $('job-identity').hidden = true;
  taskStatus('uploading', '正在上传照片；服务返回任务 ID 后会自动保存。');
  overlay('正在上传照片', '上传完成后，任务将进入重建队列。', true);
  updateControls();
  try {
    const job = await api.submit(state.files.map((entry) => entry.file), state.capabilities.template.templateId,
      { signal: operation.signal });
    if (!current(operation)) return;
    state.submitting = false;
    await acceptJob(job, operation);
  } catch (error) {
    if (!current(operation) || error.name === 'AbortError') return;
    let message = errorMessage(error);
    if (!(error instanceof ApiError) || error.status >= 500 || error.status === 0) {
      message += ' 上传响应中断时，服务可能已创建任务；请先检查服务记录，避免重复提交。';
    }
    taskStatus('interrupted', message);
    overlay('本次提交未获得任务 ID', message);
    showError(error);
  } finally { if (current(operation)) { state.submitting = false; updateControls(); } }
});
$('resume').addEventListener('click', () => resume($('resume-id').value.trim()));
$('resume-id').addEventListener('keydown', (event) => { if (event.key === 'Enter' && !$('resume').disabled) resume(event.target.value.trim()); });
$('retry').addEventListener('click', () => resume(state.job?.jobId || $('resume-id').value.trim()));
$('copy-job').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(state.job.jobId); notice('任务 ID 已复制。'); }
  catch { notice(`请复制任务 ID：${state.job?.jobId || ''}`); }
});
$('settings').addEventListener('click', () => $('connection-dialog').showModal());
$('close-settings').addEventListener('click', () => $('connection-dialog').close());
$('connection-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (state.connecting) return;
  try { api.setToken($('api-token').value.trim()); }
  catch (error) { $('connection-error').textContent = error.message; return; }
  newOperation(); // A previous token must not leave active requests updating this connection.
  state.loading = false;
  $('api-token').value = '';
  if (await connect()) {
    const jobId = state.job?.jobId || $('resume-id').value.trim();
    if (JOB_ID.test(jobId)) await resume(jobId);
  } else if (state.job) {
    taskStatus('interrupted', '连接未成功。重新连接后可继续读取已保存的任务。');
    $('retry').hidden = false;
  }
});
all('[data-mode]').forEach((button) => button.addEventListener('click', () => {
  viewer?.setMode(button.dataset.mode);
  all('[data-mode]').forEach((option) => option.setAttribute('aria-pressed', String(option === button)));
}));
all('[data-view]').forEach((button) => button.addEventListener('click', () => {
  $('auto-rotate').checked = false;
  viewer?.setAutoRotate(false);
  viewer?.setView(button.dataset.view);
}));
$('reset-view').addEventListener('click', () => {
  $('auto-rotate').checked = false; viewer?.setAutoRotate(false); viewer?.reset();
});
$('auto-rotate').addEventListener('change', (event) => viewer?.setAutoRotate(event.target.checked));
$('exposure').addEventListener('input', (event) => {
  const value = Number(event.target.value);
  $('exposure-value').textContent = value.toFixed(2);
  viewer?.setExposure(value);
});
$('screenshot').addEventListener('click', async () => {
  try {
    const data = viewer.screenshot();
    const bytes = Uint8Array.from(atob(data.split(',')[1]), (char) => char.charCodeAt(0));
    saveBlob(new Blob([bytes], { type: 'image/png' }), `face-${state.job.jobId}-preview.png`);
  } catch (error) { showError(error); }
});
$('download').addEventListener('click', async () => {
  if ($('download').disabled) return;
  const job = state.job;
  const controller = new AbortController();
  state.downloadController = controller;
  state.downloading = true;
  updateControls();
  try {
    const blob = await api.download(job, { signal: controller.signal });
    if (controller.signal.aborted) return;
    saveBlob(blob, `face-${job.jobId}.zip`);
    notice('资产 ZIP 校验通过，已交给浏览器保存。');
  } catch (error) { if (!controller.signal.aborted) showError(error); }
  finally {
    if (state.downloadController === controller) {
      state.downloadController = null; state.downloading = false; updateControls();
    }
  }
});
$('dismiss-notice').addEventListener('click', () => { $('notice').hidden = true; });
window.addEventListener('pagehide', (event) => {
  if (event.persisted) return;
  clearTimeout(state.timer); state.controller?.abort(); viewer?.dispose();
  state.downloadController?.abort();
  state.files.forEach((entry) => URL.revokeObjectURL(entry.url));
});

let savedJob = new URL(location.href).searchParams.get('job');
if (!savedJob) { try { savedJob = localStorage.getItem(STORAGE_KEY); } catch { /* Optional storage. */ } }
if (savedJob && JOB_ID.test(savedJob)) $('resume-id').value = savedJob;
renderFiles();
if (await connect()) {
  if (savedJob && JOB_ID.test(savedJob)) await resume(savedJob);
} else if ($('connection').textContent === '需要令牌') $('connection-dialog').showModal();
