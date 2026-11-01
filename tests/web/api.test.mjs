import assert from 'node:assert/strict';
import { createHash, webcrypto } from 'node:crypto';
import { test } from 'node:test';
import { ApiError, BrowserClient } from '../../src/faceAppearance/web/api.js';

const ID = '0123456789abcdef0123456789abcdef';
const OTHER_ID = 'a'.repeat(32);
const PATH = `/v1/jobs/${ID}`;
const payload = new TextEncoder().encode('verified asset bytes');
const sha256 = createHash('sha256').update(payload).digest('hex');
const metadata = { url: `${PATH}/download`, sha256, sizeBytes: payload.byteLength };
const job = (overrides = {}) => ({
  apiVersion: '1', jobId: ID, status: 'succeeded', statusUrl: PATH, download: metadata, ...overrides,
});
const json = (value, options) => new Response(JSON.stringify(value), {
  headers: { 'Content-Type': 'application/json' }, ...options,
});

test('requests stay on this origin and authentication changes only in memory', async (t) => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url, options });
    return json({ apiVersion: '1', ready: true });
  });
  const client = new BrowserClient({ token: 'first-token' });
  assert.equal((await client.getCapabilities()).ready, true);
  client.setToken('second-token');
  await client.getCapabilities();
  client.setToken('');
  await client.getCapabilities();
  assert.equal(calls[0].url, '/v1/capabilities');
  assert.equal(calls[0].options.headers.Authorization, 'Bearer first-token');
  assert.equal(calls[1].options.headers.Authorization, 'Bearer second-token');
  assert.equal(calls[2].options.headers.Authorization, undefined);
  assert.equal(calls[0].options.redirect, 'error');
  assert.equal(calls[0].options.credentials, 'same-origin');
  assert.equal(calls[0].options.cache, 'no-store');
  assert.equal(calls[0].options.signal.aborted, false);
  assert.deepEqual(Object.keys(client), []);
  assert.throws(() => client.setToken('token\nsecret'), TypeError);
});

test('unsafe paths, unsupported methods and malformed job IDs never reach fetch', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => { throw new Error('Must not fetch'); });
  const client = new BrowserClient({ token: 'secret' });
  for (const path of [
    'https://example.com/v1/capabilities', '//example.com/v1/capabilities',
    '/v1/../private', '/v1/capabilities?url=evil', `${PATH}/textures/../download`,
    `${PATH}/textures/01`, `${PATH}/download#fragment`, '/v1/jobs',
  ]) {
    await assert.rejects(client.request(path), { code: 'invalid_response' });
  }
  await assert.rejects(client.request(PATH, { method: 'POST' }), { code: 'invalid_response' });
  await assert.rejects(client.request(PATH, { method: 'DELETE' }), { code: 'invalid_response' });
  for (const id of ['../capabilities', 'A'.repeat(32), ID + '?x=1', '', null]) {
    await assert.rejects(client.getJob(id), { code: 'invalid_response' });
  }
  assert.equal(fetch.mock.callCount(), 0);
});

test('submission uses one multipart POST with files and template identity', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => json(job({ status: 'queued', download: null })));
  const image = new File([payload], 'face.png', { type: 'image/png' });
  const result = await new BrowserClient().submit([image], 'legacy-v1');
  assert.equal(result.status, 'queued');
  assert.equal(fetch.mock.callCount(), 1);
  const [url, options] = fetch.mock.calls[0].arguments;
  assert.equal(url, '/v1/jobs');
  assert.equal(options.method, 'POST');
  assert.equal(options.headers['Content-Type'], undefined);
  assert.equal(options.body.get('templateId'), 'legacy-v1');
  assert.equal(options.body.getAll('images').length, 1);
  assert.equal(options.body.get('images').name, 'face.png');
});

test('ambiguous POST failures are never automatically retried', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => { throw new TypeError('Network disconnected'); });
  await assert.rejects(new BrowserClient().submit([new Blob([payload])], 'legacy-v1'), /disconnected/);
  assert.equal(fetch.mock.callCount(), 1);
});

test('HTTP API errors preserve status, retry hint and server message', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => json({ apiVersion: '1', error: {
    code: 'queue_full', message: 'Queue is full', retryable: true,
  } }, { status: 429, headers: { 'Retry-After': '3' } }));
  await assert.rejects(new BrowserClient().getCapabilities(), (error) => {
    assert.ok(error instanceof ApiError);
    assert.equal(error.status, 429);
    assert.equal(error.code, 'queue_full');
    assert.equal(error.message, 'Queue is full');
    assert.equal(error.retryAfter, 3);
    assert.equal(error.retryable, true);
    return true;
  });
});

test('HTML error response retains useful HTTP error instead of JSON parser failure', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response('<h1>Unavailable</h1>', { status: 503 }));
  await assert.rejects(new BrowserClient().getCapabilities(), { status: 503, code: 'http_error' });
});

test('invalid JSON, incompatible versions and another job are rejected', async (t) => {
  const cases = [
    () => new Response('not json'),
    () => json([]),
    () => json(job({ apiVersion: 1 })),
    () => json(job({ apiVersion: '2' })),
    () => json(job({ jobId: OTHER_ID })),
    () => json(job({ status: 'mystery' })),
    () => json(job({ statusUrl: `/v1/jobs/${OTHER_ID}` })),
  ];
  for (const response of cases) {
    const mock = t.mock.method(globalThis, 'fetch', async () => response());
    await assert.rejects(new BrowserClient().getJob(ID), { code: 'invalid_response' });
    mock.mock.restore();
  }
});

test('preview schema, job identity and texture paths are checked before use', async (t) => {
  const client = new BrowserClient();
  const valid = { schemaVersion: 1, jobId: ID, materials: [{ texture: { ...metadata, url: `${PATH}/textures/0` } }] };
  const mock = t.mock.method(globalThis, 'fetch', async () => json(valid));
  assert.deepEqual(await client.getPreview(ID), valid);
  mock.mock.restore();
  for (const preview of [
    { ...valid, schemaVersion: 2 }, { ...valid, jobId: OTHER_ID },
    { ...valid, materials: undefined }, { ...valid, materials: [null] },
    { ...valid, materials: [{ texture: { ...metadata, url: `/v1/jobs/${OTHER_ID}/textures/0` } }] },
    { ...valid, materials: [{ texture: { ...metadata, url: 'https://example.com/image.png' } }] },
  ]) {
    const next = t.mock.method(globalThis, 'fetch', async () => json(preview));
    await assert.rejects(client.getPreview(ID), { code: 'invalid_response' });
    next.mock.restore();
  }
});

test('ZIP and texture downloads verify bytes and SHA-256 before returning a Blob', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(payload, { headers: { 'Content-Type': 'image/png' } }));
  const client = new BrowserClient();
  const zip = await client.download(job());
  assert.ok(zip instanceof Blob);
  assert.deepEqual(new Uint8Array(await zip.arrayBuffer()), payload);
  const texture = await client.getTexture({ ...metadata, url: `${PATH}/textures/12` });
  assert.equal(texture.type, 'image/png');
  assert.deepEqual(new Uint8Array(await texture.arrayBuffer()), payload);
});

test('unsafe download metadata, mismatched job URLs and unfinished jobs never reach fetch', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => { throw new Error('Must not fetch'); });
  const client = new BrowserClient();
  for (const download of [
    null, { ...metadata, url: `https://example.com${metadata.url}` },
    { ...metadata, url: `/v1/jobs/${OTHER_ID}/download` },
    { ...metadata, sizeBytes: 0 }, { ...metadata, sizeBytes: 256 * 1024 * 1024 + 1 },
    { ...metadata, sizeBytes: 1.5 }, { ...metadata, sha256: 'A'.repeat(64) },
  ]) {
    await assert.rejects(client.download(job({ download })), { code: 'invalid_response' });
  }
  await assert.rejects(client.download(job({ status: 'running' })), { code: 'invalid_response' });
  await assert.rejects(client.getTexture({ ...metadata, url: 'https://example.com/image.png' }), { code: 'invalid_response' });
  assert.equal(fetch.mock.callCount(), 0);
});

test('truncated download and hash mismatch are rejected', async (t) => {
  for (const responsePayload of [payload.slice(1), new Uint8Array(payload.length)]) {
    const mock = t.mock.method(globalThis, 'fetch', async () => new Response(responsePayload));
    await assert.rejects(new BrowserClient().download(job()), { code: 'invalid_response' });
    mock.mock.restore();
  }
});

test('oversized streamed download is cancelled as soon as it exceeds declared size', async (t) => {
  let cancelled = false;
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(payload);
      controller.enqueue(new Uint8Array([1]));
    },
    cancel() { cancelled = true; },
  });
  t.mock.method(globalThis, 'fetch', async () => new Response(stream));
  await assert.rejects(new BrowserClient().download(job()), { code: 'invalid_response' });
  assert.equal(cancelled, true);
});

test('oversized Content-Length aborts the download before reading bytes', async (t) => {
  let cancelled = false;
  t.mock.method(globalThis, 'fetch', async () => new Response(new ReadableStream({
    cancel() { cancelled = true; },
  }), { headers: { 'Content-Length': String(payload.length + 1) } }));
  await assert.rejects(new BrowserClient().download(job()), { code: 'invalid_response' });
  assert.equal(cancelled, true);
});

test('cancelled requests reject with AbortError and abort underlying fetch', async (t) => {
  let requestSignal;
  t.mock.method(globalThis, 'fetch', async (_url, options) => {
    requestSignal = options.signal;
    return new Promise(() => {});
  });
  const controller = new AbortController();
  const pending = new BrowserClient().getJob(ID, { signal: controller.signal });
  const rejected = assert.rejects(pending, { name: 'AbortError' });
  controller.abort();
  await rejected;
  assert.equal(requestSignal.aborted, true);
});

test('already cancelled requests do not fetch', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => { throw new Error('Must not fetch'); });
  await assert.rejects(new BrowserClient().getCapabilities({ signal: AbortSignal.abort() }), { name: 'AbortError' });
  assert.equal(fetch.mock.callCount(), 0);
});

test('60-second deadline rejects with TimeoutError even if fetch stalls', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let requestSignal;
  t.mock.method(globalThis, 'fetch', async (_url, options) => {
    requestSignal = options.signal;
    return new Promise(() => {});
  });
  const pending = new BrowserClient().getCapabilities();
  const rejected = assert.rejects(pending, { name: 'TimeoutError' });
  t.mock.timers.tick(60_000);
  await rejected;
  assert.equal(requestSignal.aborted, true);
});

test('request deadline covers body streaming after response headers', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  t.mock.method(globalThis, 'fetch', async () => new Response(new ReadableStream()));
  const pending = new BrowserClient().getCapabilities();
  const rejected = assert.rejects(pending, { name: 'TimeoutError' });
  await Promise.resolve();
  t.mock.timers.tick(60_000);
  await rejected;
});

test('missing secure-context WebCrypto rejects before download', async (t) => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: {} });
  t.after(() => Object.defineProperty(globalThis, 'crypto', descriptor || { configurable: true, value: webcrypto }));
  const fetch = t.mock.method(globalThis, 'fetch', async () => { throw new Error('Must not fetch'); });
  await assert.rejects(new BrowserClient().download(job()), /HTTPS/);
  assert.equal(fetch.mock.callCount(), 0);
});
