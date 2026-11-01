/** Same-origin HTTP client. Tokens are held in memory; submissions are never retried. */

const JOB_ID = /^[0-9a-f]{32}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const JOB_ROUTE = /^\/v1\/jobs\/([0-9a-f]{32})(?:\/(preview|download|textures\/(?:0|[1-9][0-9]*)))?$/;
const JOB_STATUSES = new Set(['queued', 'running', 'succeeded', 'failed']);
const REQUEST_TIMEOUT_MS = 60_000;
const MAX_JSON_BYTES = 16 * 1024 * 1024;
const MAX_ERROR_BYTES = 2 * 1024 * 1024;
const MAX_BLOB_BYTES = 256 * 1024 * 1024;

export class ApiError extends Error {
  constructor(status, code, message, retryAfter = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.retryAfter = retryAfter;
    this.retryable = [429, 502, 503, 504].includes(status);
  }
}

function invalid(message) {
  return new ApiError(0, 'invalid_response', message);
}

function object(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function checkedId(value) {
  if (typeof value !== 'string' || !JOB_ID.test(value)) {
    throw invalid('任务 ID 必须是 32 位小写十六进制字符串。');
  }
  return value;
}

function checkedRoute(path, method) {
  if (typeof path !== 'string' || !['GET', 'POST'].includes(method)) {
    throw invalid('请求地址或方法无效。');
  }
  const permitted = method === 'POST'
    ? path === '/v1/jobs'
    : path === '/v1/capabilities' || JOB_ROUTE.test(path);
  if (!permitted) throw invalid('请求地址必须属于当前服务的客户端 API。');
}

function checkedJob(record, expectedId) {
  if (!object(record) || record.apiVersion !== '1') throw invalid('服务返回的 API 版本不受支持。');
  const id = checkedId(record.jobId);
  if (expectedId !== undefined && id !== expectedId) throw invalid('服务返回了其他任务的数据。');
  if (!JOB_STATUSES.has(record.status)) throw invalid('服务返回了未知的任务状态。');
  if (record.statusUrl !== undefined && record.statusUrl !== `/v1/jobs/${id}`) {
    throw invalid('任务查询地址与任务 ID 不匹配。');
  }
  return record;
}

async function readBytes(response, maximum) {
  const lengthHeader = response.headers.get('Content-Length');
  if (lengthHeader !== null && /^\d+$/.test(lengthHeader) && Number(lengthHeader) > maximum) {
    await response.body?.cancel().catch(() => {});
    throw invalid('服务响应超过允许的大小。');
  }
  if (!response.body) return new Uint8Array();
  const reader = response.body.getReader();
  const chunks = [];
  let total = 0;
  let finished = false;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        finished = true;
        break;
      }
      total += value.byteLength;
      if (total > maximum) throw invalid('下载内容超过声明的大小或浏览器限制。');
      chunks.push(value);
    }
    const result = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return result;
  } finally {
    if (!finished) await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

function parseJson(bytes) {
  try {
    const value = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
    if (!object(value)) throw new Error('Expected an object');
    return value;
  } catch {
    throw invalid('服务未返回有效的 JSON 对象。');
  }
}

function retryDelay(response) {
  const value = response.headers.get('Retry-After');
  if (value === null || !/^\d+(?:\.\d+)?$/.test(value)) return null;
  const seconds = Number(value);
  return Number.isFinite(seconds) ? seconds : null;
}

export class BrowserClient {
  #token = '';

  constructor({ token = '' } = {}) {
    this.setToken(token);
  }

  setToken(token) {
    if (typeof token !== 'string' || /[\u0000-\u0020\u007f-\uffff]/.test(token)) {
      throw new TypeError('访问令牌必须是单行、无空格的 ASCII 字符串。');
    }
    this.#token = token;
  }

  async #operation(path, { method = 'GET', body, signal, accept = 'application/json' }, consume) {
    checkedRoute(path, method);
    if (signal?.aborted) throw new DOMException('请求已取消。', 'AbortError');
    const controller = new AbortController();
    let abortReason;
    let rejectAbort;
    const aborted = new Promise((_, reject) => { rejectAbort = reject; });
    const abort = (name, message) => {
      if (abortReason) return;
      abortReason = new DOMException(message, name);
      controller.abort(abortReason);
      rejectAbort(abortReason);
    };
    const onAbort = () => abort('AbortError', '请求已取消。');
    signal?.addEventListener('abort', onAbort, { once: true });
    const timer = setTimeout(() => abort('TimeoutError', '请求超过 60 秒，请稍后重试。'), REQUEST_TIMEOUT_MS);
    const headers = { Accept: accept };
    if (this.#token) headers.Authorization = `Bearer ${this.#token}`;
    try {
      return await Promise.race([aborted, (async () => {
        const response = await fetch(path, {
          method, body, signal: controller.signal, headers,
          credentials: 'same-origin', cache: 'no-store', redirect: 'error',
        });
        if (response.redirected) throw invalid('服务重定向被拒绝。');
        if (!response.ok) {
          let detail = {};
          try { detail = parseJson(await readBytes(response, MAX_ERROR_BYTES)).error; } catch { /* Use HTTP status. */ }
          if (!object(detail)) detail = {};
          const error = new ApiError(response.status,
            typeof detail.code === 'string' ? detail.code : 'http_error',
            typeof detail.message === 'string' ? detail.message : `请求失败（HTTP ${response.status}）。`,
            retryDelay(response));
          if (typeof detail.retryable === 'boolean') error.retryable = detail.retryable;
          throw error;
        }
        return consume(response);
      })()]);
    } catch (error) {
      if (abortReason) throw abortReason;
      throw error;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
    }
  }

  async request(path, { method = 'GET', body, signal } = {}) {
    return this.#operation(path, { method, body, signal }, async (response) => {
      const data = parseJson(await readBytes(response, MAX_JSON_BYTES));
      const isPreview = /^\/v1\/jobs\/[0-9a-f]{32}\/preview$/.test(path);
      if (isPreview ? data.schemaVersion !== 1 : data.apiVersion !== '1') {
        throw invalid(isPreview ? '预览数据的版本不受支持。' : '服务返回的 API 版本不受支持。');
      }
      return data;
    });
  }

  getCapabilities({ signal } = {}) {
    return this.request('/v1/capabilities', { signal });
  }

  async submit(files, templateId, { signal } = {}) {
    if (typeof templateId !== 'string' || !templateId || /[\r\n\0]/.test(templateId)) {
      throw new TypeError('请选择有效的服务模板。');
    }
    const images = Array.from(files ?? []);
    if (!images.length || images.some((file) => !(file instanceof Blob) || file.size === 0)) {
      throw new TypeError('请选择非空的照片文件。');
    }
    const body = new FormData();
    body.append('templateId', templateId);
    for (const [index, file] of images.entries()) {
      body.append('images', file, file.name || `image${index}.image`);
    }
    return checkedJob(await this.request('/v1/jobs', { method: 'POST', body, signal }));
  }

  async getJob(jobId, { signal } = {}) {
    checkedId(jobId);
    return checkedJob(await this.request(`/v1/jobs/${jobId}`, { signal }), jobId);
  }

  async getPreview(jobId, { signal } = {}) {
    checkedId(jobId);
    const preview = await this.request(`/v1/jobs/${jobId}/preview`, { signal });
    if (preview.jobId !== jobId) throw invalid('预览数据与当前任务不匹配。');
    if (!Array.isArray(preview.materials)) throw invalid('预览数据缺少材质列表。');
    for (const material of preview.materials) {
      if (!object(material)) throw invalid('预览材质无效。');
      if (material.texture === null || material.texture === undefined) continue;
      const texture = material.texture;
      const match = object(texture) && typeof texture.url === 'string' && JOB_ROUTE.exec(texture.url);
      if (!match || match[1] !== jobId || !match[2]?.startsWith('textures/')) {
        throw invalid('预览贴图地址与当前任务不匹配。');
      }
    }
    return preview;
  }

  async #blob(metadata, expectedUrl, { signal } = {}) {
    if (!object(metadata) || metadata.url !== expectedUrl || !SHA256.test(metadata.sha256)
        || !Number.isSafeInteger(metadata.sizeBytes) || metadata.sizeBytes <= 0
        || metadata.sizeBytes > MAX_BLOB_BYTES) {
      throw invalid('下载地址、文件大小或 SHA-256 无效（上限 256 MiB）。');
    }
    if (!globalThis.crypto?.subtle) {
      throw invalid('浏览器需要 HTTPS 或 localhost 才能校验下载文件。');
    }
    return this.#operation(expectedUrl, { signal, accept: 'application/octet-stream' }, async (response) => {
      const bytes = await readBytes(response, metadata.sizeBytes);
      if (bytes.byteLength !== metadata.sizeBytes) throw invalid('下载文件的大小与服务声明不一致。');
      const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
      const hex = Array.from(digest, (value) => value.toString(16).padStart(2, '0')).join('');
      if (hex !== metadata.sha256) throw invalid('下载文件的 SHA-256 校验失败。');
      return new Blob([bytes], { type: response.headers.get('Content-Type') || 'application/octet-stream' });
    });
  }

  getTexture(texture, { signal } = {}) {
    const match = object(texture) && typeof texture.url === 'string' && JOB_ROUTE.exec(texture.url);
    if (!match || !match[2]?.startsWith('textures/')) {
      return Promise.reject(invalid('贴图地址必须属于当前服务的任务贴图接口。'));
    }
    return this.#blob(texture, texture.url, { signal });
  }

  download(job, { signal } = {}) {
    try {
      checkedJob(job);
      if (job.status !== 'succeeded') throw invalid('任务完成后才能下载资产。');
      return this.#blob(job.download, `/v1/jobs/${job.jobId}/download`, { signal });
    } catch (error) {
      return Promise.reject(error);
    }
  }
}
