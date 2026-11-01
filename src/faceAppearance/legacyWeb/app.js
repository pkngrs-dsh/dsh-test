import { FaceViewer } from '/viewer-common/scene.js';

const $ = (id) => document.getElementById(id);
const all = (selector) => [...document.querySelectorAll(selector)];
const TASK_ID = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;
const labels = { pending: '排队中', processing: '重建中', completed: '已完成', failed: '失败',
  uploading: '上传中', submitting: '提交中', downloading: '下载中', interrupted: '连接中断', loading: '读取中', idle: '未开始' };
const state = { files: [], samples: [], inputMode: 'upload', health: null, connecting: false,
  job: null, submitting: false, loading: false, previewLoaded: false, generation: 0,
  controller: null, timer: null, pollFailures: 0, terminalPolls: 0, sourceImages: [], sourceKey: '', sourceIndex: 0, jobs: [] };
let viewer;
let viewerError = null;

function notice(message) {
  $('notice-text').textContent = message;
  $('notice').hidden = false;
}

function message(error) {
  if (error?.name === 'TimeoutError') return '请求超时，可以继续读取同一任务。';
  if (error instanceof TypeError && /fetch|network/i.test(error.message)) return '暂时无法连接本地预览服务，请确认服务仍在运行。';
  const known = { queue_full: '远端队列已满，请稍后提交。', service_restarted: '服务曾在任务完成前重启，本次任务已结束。',
    reconstruction_failed: '重建未能完成，请查看服务器日志中的任务记录。', not_found: '没有找到此任务，请核对任务 ID。',
    result_not_ready: '资产仍在准备中，请稍后继续读取。' };
  if (typeof error === 'string') return known[error] || error;
  return known[error?.code] || error?.message || '请求未能完成，请稍后重试。';
}

async function request(path, { signal, timeout = 30000, ...options } = {}) {
  const controller = new AbortController();
  const abort = () => controller.abort(signal.reason);
  if (signal?.aborted) abort();
  else signal?.addEventListener('abort', abort, { once: true });
  const timer = setTimeout(() => controller.abort(new DOMException('请求超时', 'TimeoutError')), timeout);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal, credentials: 'same-origin', cache: 'no-store' });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      const detail = body.detail || body;
      const error = new Error(typeof detail === 'string' ? detail : detail.message || detail.error?.message || detail.error || `请求失败（HTTP ${response.status}）`);
      error.status = response.status;
      error.code = detail.code || detail.error?.code;
      error.taskId = detail.task_id || body.task_id;
      throw error;
    }
    return response;
  } catch (error) {
    if (controller.signal.aborted) throw controller.signal.reason || error;
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}

async function json(path, options) { return (await request(path, options)).json(); }
function jobPath(id) { return `/api/jobs/${encodeURIComponent(id)}`; }
function selectedSample() { return state.samples.find((sample) => sample.id === $('sample-select').value); }
function chosenImages() {
  return state.inputMode === 'sample' ? selectedSample()?.images || []
    : state.files.map((entry) => ({ name: entry.file.name, url: entry.url }));
}
function activeJob() { return ['pending', 'processing'].includes(state.job?.status); }
function canSend() { return Boolean(state.health && state.health.status === 'ok' && state.health.accepting_jobs !== false && state.health.previewReady !== false && state.health.models !== 'loading'); }
function readinessMessage() { return state.health?.previewMessage || '服务就绪后可提交；已有任务仍可继续读取。'; }
function displayedStatus(job) {
  if (job?.status === 'failed') return 'failed';
  if (job?.status === 'completed' && job.assetReady) return 'completed';
  return { uploading: 'uploading', submitting: 'submitting', queued: 'pending', reconstructing: 'processing',
    downloading: 'downloading', preparing_preview: 'downloading', completed: 'downloading', failed: 'failed' }[job?.phase]
    || (job?.status === 'pending' ? 'submitting' : job?.status || 'idle');
}
function prettySeconds(seconds) {
  if (!Number.isFinite(Number(seconds)) || seconds === null || seconds === undefined) return '—';
  const whole = Math.max(0, Math.floor(Number(seconds)));
  return whole < 60 ? `${whole} 秒` : `${Math.floor(whole / 60)} 分 ${String(whole % 60).padStart(2, '0')} 秒`;
}

function updateControls() {
  const count = chosenImages().length;
  const active = activeJob();
  $('submit').disabled = state.submitting || state.connecting || active || !canSend() || count < 4 || count > 16;
  $('submit').querySelector('span').textContent = state.submitting ? '正在提交照片…'
    : active ? ({ uploading: '正在传送照片…', submitting: '正在提交任务…', pending: '正在等待远端 GPU…' }[displayedStatus(state.job)] || '远端正在处理…') : '发送到服务器重建';
  $('submit-hint').textContent = state.submitting ? '正在上传并创建任务，请保持页面打开。'
    : active ? '自动查询同一任务，无需再次提交。'
    : !canSend() ? readinessMessage()
    : count < 4 ? `还需选择 ${4 - count} 张照片。` : `共 ${count} 张照片，将使用${$('gender').value === 'male' ? '男性' : '女性'}模板。`;
  for (const id of ['photos', 'clear-files', 'gender', 'sample-select', 'upload-tab', 'sample-tab', 'resume', 'retry']) $(id).disabled = state.submitting;
  $('reconnect').disabled = state.connecting;
  $('dropzone').dataset.disabled = String(state.submitting);
  all('#thumbnails button, .recent-job').forEach((button) => { button.disabled = state.submitting; });
  $('download').disabled = state.job?.status !== 'completed' || !state.job?.assetReady;
  all('[data-mode], [data-view], #reset-view, #auto-rotate, #exposure, #screenshot').forEach((control) => {
    control.disabled = !state.previewLoaded || Boolean(viewerError);
  });
}

function overlay(title, description, loading = false) {
  $('viewport-overlay').hidden = false;
  $('viewport-overlay').classList.toggle('loading', loading);
  $('overlay-title').textContent = title;
  $('overlay-message').textContent = description;
}

try {
  viewer = new FaceViewer($('viewport'), { onError(error) {
    viewerError = error;
    state.previewLoaded = false;
    overlay('三维预览暂不可用', message(error));
    updateControls();
  } });
} catch (error) {
  viewerError = error;
  overlay('浏览器未能启用 WebGL', '请开启浏览器硬件加速。你仍可提交任务和下载资产。');
}

function safeImageURL(url) {
  if (state.files.some((entry) => entry.url === url)) return url;
  const parsed = new URL(url, location.origin);
  if (parsed.origin !== location.origin || !parsed.pathname.startsWith('/api/')) throw new Error('照片或贴图地址不属于本地预览服务。');
  return parsed.href;
}

function selectSource(index) {
  if (!state.sourceImages[index]) return;
  state.sourceIndex = index;
  const entry = state.sourceImages[index];
  $('source-image').src = safeImageURL(entry.url);
  $('source-image').alt = `源照片：${entry.name}`;
  $('source-name').textContent = entry.name;
  $('source-name').title = entry.name;
  $('source-count').textContent = `${index + 1} / ${state.sourceImages.length}`;
  all('#source-thumbnails button').forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
}

function renderSource(images, caption = '本次任务的源照片，可对照正面与侧面细节。') {
  const key = JSON.stringify(images.map((entry) => [entry.name, entry.url]));
  $('source-caption').textContent = caption;
  if (key === state.sourceKey) return;
  state.sourceKey = key;
  state.sourceImages = images;
  state.sourceIndex = 0;
  $('source-panel').hidden = !images.length;
  $('comparison').classList.toggle('has-source', Boolean(images.length));
  $('source-thumbnails').replaceChildren();
  if (!images.length) { $('source-image').removeAttribute('src'); return; }
  images.forEach((entry, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.title = entry.name;
    button.setAttribute('aria-label', `对照照片 ${index + 1}：${entry.name}`);
    button.setAttribute('aria-pressed', 'false');
    const image = document.createElement('img');
    image.src = safeImageURL(entry.url);
    image.alt = entry.name;
    button.append(image);
    button.addEventListener('click', () => selectSource(index));
    $('source-thumbnails').append(button);
  });
  selectSource(0);
}

function renderInputs() {
  const images = chosenImages();
  $('thumbnails').replaceChildren();
  images.forEach((entry, index) => {
    const item = document.createElement('div');
    item.className = 'thumbnail';
    const view = document.createElement('button');
    view.type = 'button';
    view.className = 'photo-select';
    view.setAttribute('aria-label', `查看 ${entry.name}`);
    const img = document.createElement('img');
    img.src = safeImageURL(entry.url);
    img.alt = entry.name;
    view.append(img);
    view.addEventListener('click', () => {
      renderSource(images, '待提交的照片。提交后将用于远端重建。');
      selectSource(index);
    });
    item.append(view);
    if (state.inputMode === 'upload') {
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.textContent = '×';
      remove.setAttribute('aria-label', `移除 ${entry.name}`);
      remove.addEventListener('click', () => {
        if (state.submitting) return;
        URL.revokeObjectURL(state.files[index].url);
        state.files.splice(index, 1);
        renderInputs();
      });
      item.append(remove);
    }
    $('thumbnails').append(item);
  });
  $('file-count').textContent = `${images.length} / 16`;
  $('files-toolbar').hidden = state.inputMode !== 'upload' || !state.files.length;
  $('file-size').textContent = `共 ${(state.files.reduce((sum, entry) => sum + entry.file.size, 0) / 1024 ** 2).toFixed(1)} MiB`;
  if (!state.job) renderSource(images, '待提交的照片。提交后将用于远端重建。');
  updateControls();
}

function addFiles(files) {
  if (state.submitting) return;
  const skipped = [];
  let total = state.files.reduce((sum, entry) => sum + entry.file.size, 0);
  for (const file of files) {
    if (state.files.some((entry) => entry.file.name === file.name && entry.file.size === file.size && entry.file.lastModified === file.lastModified)) continue;
    if (!/\.(jpe?g|png|webp|bmp|tiff?)$/i.test(file.name)) { skipped.push(`${file.name}：不支持的图片格式`); continue; }
    if (!file.size || file.size > 25 * 1024 ** 2) { skipped.push(`${file.name}：单张图片需小于 25 MiB`); continue; }
    if (state.files.length >= 16) { skipped.push('最多选择 16 张照片'); break; }
    if (total + file.size > 200 * 1024 ** 2) { skipped.push('照片总大小不能超过 200 MiB'); continue; }
    state.files.push({ file, url: URL.createObjectURL(file) });
    total += file.size;
  }
  renderInputs();
  if (skipped.length) notice(skipped.slice(0, 3).join('；'));
}

function setInputMode(mode) {
  if (state.submitting) return;
  state.inputMode = mode;
  $('upload-tab').setAttribute('aria-pressed', String(mode === 'upload'));
  $('sample-tab').setAttribute('aria-pressed', String(mode === 'sample'));
  $('upload-input').hidden = mode !== 'upload';
  $('sample-input').hidden = mode !== 'sample';
  if (mode === 'sample' && selectedSample()?.gender) $('gender').value = selectedSample().gender;
  renderInputs();
}

function taskStatus(status, description) {
  $('task-pill').textContent = labels[status] || status;
  $('task-pill').dataset.state = status;
  $('task-message').textContent = description;
  const progress = { uploading: 0, submitting: 0, pending: 1, processing: 1, downloading: 2, completed: 3, failed: 1 }[status] ?? -1;
  all('.task-progress i').forEach((step, index) => {
    step.classList.toggle('done', index < progress);
    step.classList.toggle('active', index === progress && ['uploading', 'submitting', 'pending', 'processing', 'downloading'].includes(status));
  });
}

function remember(id) {
  $('resume-id').value = id;
  const url = new URL(location.href);
  url.searchParams.set('job', id);
  history.replaceState(null, '', url);
}

function newOperation() {
  clearTimeout(state.timer);
  state.controller?.abort();
  state.controller = new AbortController();
  state.generation += 1;
  state.pollFailures = 0;
  state.terminalPolls = 0;
  state.loading = false;
  $('retry').hidden = true;
  return { generation: state.generation, signal: state.controller.signal };
}
function current(operation) { return operation.generation === state.generation && !operation.signal.aborted; }
function resetPreview() {
  viewer?.clear();
  state.previewLoaded = false;
  $('preview-label').textContent = '等待资产';
  $('asset-name').textContent = '尚未生成资产';
  $('asset-description').textContent = '完成后可查看模型，并下载旧版协议的 ZIP 资产包。';
  for (const id of ['source-vertices', 'render-vertices', 'face-count', 'material-count']) $(id).textContent = '—';
  $('viewport-state').textContent = '本地 Three.js 预览';
  $('auto-rotate').checked = false;
  viewer?.setAutoRotate(false);
}

async function loadPreview(operation) {
  if (state.previewLoaded || state.loading) return;
  if (!viewer || viewerError) { overlay('浏览器未能启用三维预览', '资产已经生成，可以下载 ZIP。预览需要开启浏览器硬件加速。'); return; }
  state.loading = true;
  overlay('正在载入三维资产', '读取网格与贴图，准备本地预览…', true);
  updateControls();
  try {
    const preview = await json(`${jobPath(state.job.task_id)}/preview`, { signal: operation.signal, timeout: 120000 });
    if (!current(operation)) return;
    await viewer.load(preview, async (url) => (await request(safeImageURL(url), { signal: operation.signal, timeout: 90000 })).blob());
    if (!current(operation)) return;
    state.previewLoaded = true;
    $('viewport-overlay').hidden = true;
    $('preview-label').textContent = '资产已载入';
    $('viewport-state').textContent = '远端生成 · 本地工作室光照';
    $('asset-name').textContent = preview.templateId || `${state.job.gender === 'male' ? '男性' : '女性'}头部资产`;
    $('asset-description').textContent = '旧版 OBJ / MTL / PNG 资产 · 米制坐标';
    const fields = { 'source-vertices': 'sourceVertices', 'render-vertices': 'renderVertices', 'face-count': 'faces', 'material-count': 'materials' };
    for (const [id, key] of Object.entries(fields)) $(id).textContent = Number.isFinite(preview.counts?.[key]) ? preview.counts[key].toLocaleString('zh-CN') : '—';
    if (preview.warnings?.includes('preview_contains_zero_length_normals')) notice('资产存在零长度法线，部分预览光照可能异常。');
  } catch (error) {
    if (!current(operation) || error.name === 'AbortError') return;
    overlay('三维资产暂未载入', message(error));
    $('retry').hidden = false;
    notice(message(error));
  } finally {
    if (current(operation)) { state.loading = false; updateControls(); }
  }
}

function callbackLabel(callback) {
  const status = callback?.state;
  return { received: '已收到', delivered: '已送达', succeeded: '已送达', success: '已送达', completed: '已送达', pending: '等待回调', waiting: '等待回调', failed: '回调失败', skipped: '未发送' }[status] || status || '等待回调';
}

async function acceptJob(job, operation) {
  if (!current(operation)) return;
  state.job = job;
  remember(job.task_id);
  $('job-identity').hidden = false;
  $('job-id').textContent = job.task_id;
  $('job-id').title = job.task_id;
  $('job-details').hidden = false;
  $('elapsed').textContent = prettySeconds(job.elapsedSeconds);
  $('callback-state').textContent = callbackLabel(job.callback);
  $('retry').hidden = true;
  if (Array.isArray(job.images) && job.images.length) renderSource(job.images);
  const display = displayedStatus(job);
  if (display === 'failed') {
    taskStatus('failed', message(job.error));
    overlay('本次任务未完成', message(job.error));
    $('retry').hidden = false;
  } else if (display === 'uploading') {
    taskStatus('uploading', '正在将照片传送到服务器，并准备远端可读取的图片。');
    overlay('正在传送照片', '照片准备完成后，会向远端服务提交重建请求。', true);
  } else if (display === 'submitting') {
    taskStatus('submitting', '正在向远端服务提交请求，等待任务确认。');
    overlay('正在提交重建请求', '收到远端确认后，会继续显示排队或重建进度。', true);
  } else if (display === 'pending') {
    taskStatus('pending', '任务已被接收，正在等待远端 GPU。页面会自动查询进度。');
    overlay('任务已进入远端队列', '可以关闭页面，之后通过任务 ID 继续查看。', true);
  } else if (display === 'processing') {
    taskStatus('processing', '正在远端重建几何、融合贴图并装配资产。完成后自动载入。');
    overlay('正在重建三维头部', '几何重建 → 纹理融合 → 模板装配', true);
  } else if (display === 'downloading') {
    taskStatus('downloading', '远端重建完成，正在将资产下载到本地并准备预览。');
    overlay('正在取回三维资产', '下载 ZIP → 检查资产 → 准备本地预览', true);
  } else if (display === 'completed') {
    taskStatus('completed', '资产已回到本地。可对照照片检查纹理和几何，并下载 ZIP。');
  } else {
    taskStatus('processing', '正在读取任务进度…');
  }
  updateControls();
  if (activeJob() || (job.status === 'completed' && !job.assetReady)) {
    state.timer = setTimeout(() => poll(job.task_id, operation), 2500);
  } else if (job.status === 'completed') {
    await loadPreview(operation);
    if (!current(operation)) return;
    void loadJobs();
    // Asset transfer can finish before the remote callback receipt is inspected.
    const callbackFinished = ['received', 'delivered', 'failed', 'skipped', 'completed', 'succeeded'].includes(job.callback?.state);
    if ((!callbackFinished || job.phase === 'downloading') && state.terminalPolls++ < 24) {
      state.timer = setTimeout(() => poll(job.task_id, operation), 2500);
    }
  }
}

async function poll(id, operation) {
  try {
    const job = await json(jobPath(id), { signal: operation.signal, timeout: 45000 });
    if (!current(operation)) return;
    state.pollFailures = 0;
    await acceptJob(job, operation);
  } catch (error) {
    if (!current(operation) || error.name === 'AbortError') return;
    state.pollFailures += 1;
    const retryable = !error.status || error.status >= 500 || [408, 429].includes(error.status);
    if (retryable && state.pollFailures <= 5) {
      const seconds = Math.min(2 ** state.pollFailures, 30);
      taskStatus('interrupted', `连接暂时中断，${seconds} 秒后继续查询同一任务。不会重复提交。`);
      state.timer = setTimeout(() => poll(id, operation), seconds * 1000);
    } else {
      taskStatus('interrupted', `${message(error)} 任务 ID 已保留。`);
      $('retry').hidden = false;
      notice(message(error));
    }
    updateControls();
  }
}

async function resume(id) {
  if (state.submitting) return;
  id = id.trim();
  if (!TASK_ID.test(id)) { notice('任务 ID 需为 1–128 位字母、数字、下划线或连字符。'); return; }
  const operation = newOperation();
  state.job = { task_id: id, status: 'loading' };
  remember(id);
  resetPreview();
  renderSource([]);
  $('job-identity').hidden = false;
  $('job-id').textContent = id;
  $('job-details').hidden = true;
  taskStatus('loading', '正在读取任务状态…');
  overlay('正在读取已有任务', '加载任务状态和本地资产，不会重新提交照片。', true);
  updateControls();
  await poll(id, operation);
}

async function submit() {
  if ($('submit').disabled) return;
  const operation = newOperation();
  state.submitting = true;
  state.job = null;
  const freshURL = new URL(location.href);
  freshURL.searchParams.delete('job');
  history.replaceState(null, '', freshURL);
  resetPreview();
  renderSource(chosenImages());
  $('job-identity').hidden = true;
  $('job-details').hidden = true;
  $('notice').hidden = true;
  taskStatus('uploading', '正在准备照片并创建远端任务…');
  overlay('正在提交照片', '提交确认后将自动查询同一任务。', true);
  updateControls();
  try {
    const gender = $('gender').value;
    let result;
    if (state.inputMode === 'sample') {
      result = await json(`/api/samples/${encodeURIComponent(selectedSample().id)}/jobs`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ gender }), signal: operation.signal, timeout: 180000 });
    } else {
      const body = new FormData();
      body.append('gender', gender);
      state.files.forEach((entry) => body.append('photos', entry.file, entry.file.name));
      result = await json('/api/jobs', { method: 'POST', body, signal: operation.signal, timeout: 180000 });
    }
    if (!current(operation)) return;
    if (!TASK_ID.test(result.task_id || '')) throw new Error('服务没有返回有效任务 ID，请先刷新最近任务确认提交结果。');
    state.submitting = false;
    state.job = { ...result, gender };
    remember(result.task_id);
    await poll(result.task_id, operation);
    void loadJobs();
  } catch (error) {
    if (!current(operation) || error.name === 'AbortError') return;
    if (TASK_ID.test(error.taskId || '')) {
      state.job = { task_id: error.taskId, status: 'pending' };
      remember(error.taskId);
      $('job-id').textContent = error.taskId;
      $('job-identity').hidden = false;
      $('retry').hidden = false;
    }
    const uncertain = !error.status || error.status >= 500;
    const detail = `${message(error)}${uncertain ? ' 提交结果暂未确认，请先刷新最近任务，确认是否已创建任务后再操作。' : ''}`;
    taskStatus('interrupted', detail);
    overlay('提交暂未完成', detail);
    notice(detail);
    void loadJobs();
  } finally {
    if (current(operation)) { state.submitting = false; updateControls(); }
  }
}

async function connect() {
  if (state.connecting) return;
  state.connecting = true;
  $('connection').textContent = '正在连接服务';
  $('connection').dataset.state = 'pending';
  updateControls();
  try {
    const health = await json('/api/health', { timeout: 25000 });
    state.health = health;
    $('connection').textContent = canSend() ? '远端服务就绪' : health.previewMessage || '远端正在准备';
    $('connection').dataset.state = canSend() ? 'ready' : 'pending';
    $('connection').title = health.previewMessage || '';
    $('remote-address').textContent = health.remoteBaseUrl || '远端执行 · 本地查看';
    $('remote-address').title = health.remoteBaseUrl || '';
  } catch (error) {
    state.health = null;
    $('connection').textContent = '远端连接中断';
    $('connection').dataset.state = 'error';
    $('connection').title = message(error);
  } finally { state.connecting = false; updateControls(); }
}

async function recover() {
  if (state.submitting) return;
  const id = state.job?.task_id || $('resume-id').value.trim();
  if (!TASK_ID.test(id)) return;
  $('retry').disabled = true;
  try {
    await json(`${jobPath(id)}/resume`, { method: 'POST' });
    await resume(id);
  } catch (error) {
    notice(message(error));
  } finally { updateControls(); }
}

async function loadSamples() {
  try {
    const response = await json('/api/samples');
    state.samples = response.samples || [];
    $('sample-select').replaceChildren();
    if (!state.samples.length) {
      $('sample-select').add(new Option('暂无内置样本', ''));
      $('sample-hint').textContent = '可切换到上传照片，选择本地文件。';
    } else {
      state.samples.forEach((sample) => $('sample-select').add(new Option(`${sample.label} · ${sample.images.length} 张`, sample.id)));
      $('sample-hint').textContent = '选择样本后，点击发送到服务器重建。';
    }
    if (state.inputMode === 'sample') setInputMode('sample');
  } catch (error) {
    $('sample-select').replaceChildren(new Option('样本读取失败', ''));
    $('sample-hint').textContent = message(error);
  }
}

async function loadJobs() {
  $('refresh-jobs').disabled = true;
  try {
    const response = await json('/api/jobs');
    state.jobs = response.jobs || [];
    $('recent-jobs').replaceChildren();
    if (!state.jobs.length) {
      const empty = document.createElement('p');
      empty.className = 'subtle-note';
      empty.textContent = '还没有本地联调任务。';
      $('recent-jobs').append(empty);
    }
    state.jobs.slice(0, 12).forEach((job) => {
      const button = document.createElement('button');
      button.className = 'recent-job';
      button.type = 'button';
      button.title = job.task_id;
      button.setAttribute('aria-current', String(job.task_id === state.job?.task_id));
      const information = document.createElement('span');
      const id = document.createElement('code');
      id.textContent = job.task_id;
      const date = document.createElement('small');
      const timestamp = new Date(job.createdAt);
      date.textContent = `${job.gender === 'male' ? '男性' : '女性'}模板${Number.isNaN(timestamp.getTime()) ? '' : ` · ${timestamp.toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })}`}`;
      information.append(id, date);
      const status = document.createElement('b');
      status.textContent = labels[displayedStatus(job)] || job.status;
      button.append(information, status);
      button.addEventListener('click', () => resume(job.task_id));
      $('recent-jobs').append(button);
    });
  } catch (error) {
    const text = document.createElement('p');
    text.className = 'subtle-note';
    text.textContent = message(error);
    $('recent-jobs').replaceChildren(text);
  } finally { $('refresh-jobs').disabled = false; updateControls(); }
}

$('photos').addEventListener('change', (event) => { addFiles([...event.target.files]); event.target.value = ''; });
$('dropzone').addEventListener('dragover', (event) => { event.preventDefault(); if (!state.submitting) $('dropzone').classList.add('dragover'); });
$('dropzone').addEventListener('dragleave', () => $('dropzone').classList.remove('dragover'));
$('dropzone').addEventListener('drop', (event) => { event.preventDefault(); $('dropzone').classList.remove('dragover'); addFiles([...event.dataTransfer.files]); });
$('clear-files').addEventListener('click', () => { if (state.submitting) return; state.files.forEach((entry) => URL.revokeObjectURL(entry.url)); state.files = []; renderInputs(); });
$('upload-tab').addEventListener('click', () => setInputMode('upload'));
$('sample-tab').addEventListener('click', () => setInputMode('sample'));
$('sample-select').addEventListener('change', () => { if (selectedSample()?.gender) $('gender').value = selectedSample().gender; renderInputs(); });
$('gender').addEventListener('change', updateControls);
$('submit').addEventListener('click', submit);
$('resume').addEventListener('click', () => resume($('resume-id').value));
$('resume-id').addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); void resume($('resume-id').value); } });
$('retry').addEventListener('click', recover);
$('refresh-jobs').addEventListener('click', loadJobs);
$('reconnect').addEventListener('click', () => { void connect(); void loadSamples(); void loadJobs(); });
$('dismiss-notice').addEventListener('click', () => { $('notice').hidden = true; });
$('copy-job').addEventListener('click', async () => { try { await navigator.clipboard.writeText(state.job.task_id); notice('已复制任务 ID。'); } catch { notice('复制失败，可在“打开已有任务”中手动复制任务 ID。'); } });
all('[data-mode]').forEach((button) => button.addEventListener('click', () => { viewer?.setMode(button.dataset.mode); all('[data-mode]').forEach((entry) => entry.setAttribute('aria-pressed', String(entry === button))); }));
all('[data-view]').forEach((button) => button.addEventListener('click', () => { $('auto-rotate').checked = false; viewer?.setAutoRotate(false); viewer?.setView(button.dataset.view); }));
$('reset-view').addEventListener('click', () => { $('auto-rotate').checked = false; viewer?.setAutoRotate(false); viewer?.reset(); });
$('auto-rotate').addEventListener('change', (event) => viewer?.setAutoRotate(event.target.checked));
$('exposure').addEventListener('input', (event) => { const value = Number(event.target.value); viewer?.setExposure(value); $('exposure-value').textContent = value.toFixed(2); });
$('screenshot').addEventListener('click', () => { try { const anchor = document.createElement('a'); anchor.href = viewer.screenshot(); anchor.download = `${state.job.task_id}-preview.png`; anchor.click(); } catch (error) { notice(message(error)); } });
$('download').addEventListener('click', () => { if ($('download').disabled) return; const anchor = document.createElement('a'); anchor.href = `${jobPath(state.job.task_id)}/download`; anchor.download = `${state.job.task_id}.zip`; anchor.click(); });
window.addEventListener('pagehide', () => { clearTimeout(state.timer); state.controller?.abort(); });
window.addEventListener('pageshow', (event) => { if (event.persisted && state.job?.task_id) void resume(state.job.task_id); });

renderInputs();
const initialJob = new URL(location.href).searchParams.get('job');
await Promise.allSettled([connect(), loadSamples(), loadJobs(), initialJob ? resume(initialJob) : Promise.resolve()]);
