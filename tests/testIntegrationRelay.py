"""CPU-only coverage for the opt-in deployed-service integration endpoints."""

import json
import os
from pathlib import Path
import stat
import time
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from faceAppearance.integrationRelay import registerIntegrationRelay
from faceAppearance.legacyTransport import TransportError, resolveUrl


class IntegrationRelayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(os.environ, {"FACE_APPEARANCE_INTEGRATION_ROOT": str(self.root)})
        self.environment.start()
        self.token = "a" * 64
        self.directory = self.root / self.token
        self.directory.mkdir(mode=0o700)
        (self.directory / "registration.json").write_text(json.dumps({
            "task_id": "preview_1", "inputExpiresAt": time.time() + 7200,
        }))
        app = FastAPI()
        registerIntegrationRelay(app)
        self.client = AsyncClient(transport=ASGITransport(app=app), base_url="http://integration.test")
        self.url = "/_integration/callback/" + self.token
        self.payload = {"task_id": "preview_1", "status": "completed", "sh_coeffs": [0.] * 16,
                        "result_url": "https://example.invalid/private-result?test=not-a-secret"}

    async def asyncTearDown(self):
        await self.client.aclose()
        self.environment.stop()
        self.temporary.cleanup()

    async def testOptInAndUnknownTokens(self):
        with patch.dict(os.environ, {"FACE_APPEARANCE_INTEGRATION_ROOT": ""}):
            app = FastAPI()
            registerIntegrationRelay(app)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://integration.test") as client:
                self.assertEqual((await client.post(self.url, json=self.payload)).status_code, 404)
        for token in ("x", "b" * 64, "G" * 64):
            self.assertEqual((await self.client.post("/_integration/callback/" + token,
                                             json=self.payload)).status_code, 404)

    async def testPrivateAtomicIdempotentRecord(self):
        response = await self.client.post(self.url, json=self.payload)
        self.assertEqual(response.status_code, 200)
        path = self.directory / "callback.json"
        original = path.read_bytes()
        self.assertEqual(json.loads(original)["payload"], self.payload)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual((await self.client.post(self.url, json=self.payload)).status_code, 200)
        self.assertEqual(path.read_bytes(), original)
        different = {**self.payload, "status": "failed", "result_url": "error"}
        self.assertEqual((await self.client.post(self.url, json=different)).status_code, 409)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(sorted(p.name for p in self.directory.iterdir()),
                         ["callback.json", "registration.json"])

    async def testRejectsMalformedPayload(self):
        bad = [[], {**self.payload, "task_id": "other"}, {**self.payload, "extra": 1},
               {**self.payload, "status": "processing"}, {**self.payload, "sh_coeffs": [0.] * 15},
               {**self.payload, "sh_coeffs": [True] * 16}, {**self.payload, "result_url": None}]
        for payload in bad:
            with self.subTest(payload=payload):
                self.assertEqual((await self.client.post(self.url, json=payload)).status_code, 400)
        self.assertEqual((await self.client.post(self.url, content=b"{" + b" " * (64 * 1024))).status_code, 413)
        nonFinite = json.dumps({**self.payload, "sh_coeffs": [float("nan")] * 16})
        self.assertEqual((await self.client.post(self.url, content=nonFinite)).status_code, 400)
        self.assertFalse((self.directory / "callback.json").exists())

    async def testRegistrationSymlinkIsNotFollowed(self):
        registration = self.directory / "registration.json"
        registration.unlink()
        other = self.root / "other.json"
        other.write_text('{"task_id":"preview_1"}')
        registration.symlink_to(other)
        self.assertEqual((await self.client.post(self.url, json=self.payload)).status_code, 404)

    async def testRegisteredInputDownload(self):
        inputs = self.directory / "inputs"
        inputs.mkdir(mode=0o700)
        (inputs / "00.image").write_bytes(b"private sample bytes")
        url = f"/_integration/inputs/{self.token}/0"
        response = await self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"private sample bytes")
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertEqual(response.headers["content-type"], "application/octet-stream")
        self.assertEqual(response.headers["content-length"], "20")
        self.assertFalse((self.directory / "callback.json").exists())

    async def testInputRegistrationAndIndexRequired(self):
        inputs = self.directory / "inputs"
        inputs.mkdir(mode=0o700)
        (inputs / "00.image").write_bytes(b"photo")
        for token, index in (("x", "0"), ("b" * 64, "0"), (self.token, "-1"),
                             (self.token, "16"), (self.token, "1"), (self.token, "1.0")):
            with self.subTest(token=token, index=index):
                self.assertEqual((await self.client.get(
                    f"/_integration/inputs/{token}/{index}")).status_code, 404)
        (self.directory / "registration.json").write_text('{"task_id":"../bad"}')
        self.assertEqual((await self.client.get(f"/_integration/inputs/{self.token}/0")).status_code, 404)

    async def testExpiredInputsButCallbackStillAccepted(self):
        inputs = self.directory / "inputs"
        inputs.mkdir(mode=0o700)
        (inputs / "00.image").write_bytes(b"photo")
        registration = self.directory / "registration.json"
        for expiresAt in (None, 0, time.time() - 1, "future", True):
            registration.write_text(json.dumps({"task_id": "preview_1", "inputExpiresAt": expiresAt}))
            self.assertEqual((await self.client.get(
                f"/_integration/inputs/{self.token}/0")).status_code, 404)
        self.assertEqual((await self.client.post(self.url, json=self.payload)).status_code, 200)

    async def testInputSizeAndSymlinksRejected(self):
        inputs = self.directory / "inputs"
        inputs.mkdir(mode=0o700)
        path = inputs / "00.image"
        url = f"/_integration/inputs/{self.token}/0"
        with path.open("wb") as stream:
            stream.truncate(25 * 1024 * 1024 + 1)
        self.assertEqual((await self.client.get(url)).status_code, 404)
        path.write_bytes(b"")
        self.assertEqual((await self.client.get(url)).status_code, 404)
        path.unlink()
        outside = self.root / "outside.image"
        outside.write_bytes(b"not registered")
        path.symlink_to(outside)
        self.assertEqual((await self.client.get(url)).status_code, 404)
        path.unlink()
        inputs.rmdir()
        outsideDirectory = self.root / "outside"
        outsideDirectory.mkdir()
        (outsideDirectory / "00.image").write_bytes(b"not registered")
        inputs.symlink_to(outsideDirectory, target_is_directory=True)
        self.assertEqual((await self.client.get(url)).status_code, 404)


class PublicDnsOverridesTests(unittest.TestCase):
    def testOverridePinsPublicAddressAndPreservesHost(self):
        value = json.dumps({"example.com": "8.8.8.8"})
        with patch.dict(os.environ, {"FACE_APPEARANCE_PUBLIC_DNS_OVERRIDES": value}), \
                patch("faceAppearance.legacyTransport.socket.getaddrinfo") as lookup:
            parts, host, port, ips = resolveUrl("https://example.com:8443/hello/")
        lookup.assert_not_called()
        self.assertEqual((host, port, ips), ("example.com", 8443, ["8.8.8.8"]))
        self.assertEqual(parts.path, "/hello/")

    def testListAndUnmatchedHost(self):
        value = json.dumps({"EXAMPLE.COM.": ["8.8.8.8", "1.1.1.1", "8.8.8.8"]})
        with patch.dict(os.environ, {"FACE_APPEARANCE_PUBLIC_DNS_OVERRIDES": value}):
            self.assertEqual(resolveUrl("https://example.com/x")[3], ["8.8.8.8", "1.1.1.1"])
            with patch("faceAppearance.legacyTransport.socket.getaddrinfo",
                       return_value=[(2, 1, 6, "", ("9.9.9.9", 443))]) as lookup:
                self.assertEqual(resolveUrl("https://other.example/x")[3], ["9.9.9.9"])
                lookup.assert_called_once()

    def testInvalidOrPrivateOverridesRejectedEvenWithInternalEnabled(self):
        values = ["not-json", "[]", '{"example.com":[]}', '{"example.com":"localhost"}',
                  '{"example.com":"127.0.0.1"}', '{"example.com":"192.168.1.1"}',
                  '{"example.com":"169.254.169.254"}', '{"example.com":"224.0.0.1"}',
                  '{"example.com":"::ffff:127.0.0.1"}', '{"example.com":42}']
        for value in values:
            with self.subTest(value=value), \
                    patch.dict(os.environ, {"FACE_APPEARANCE_PUBLIC_DNS_OVERRIDES": value}):
                with self.assertRaises(TransportError):
                    resolveUrl("https://example.com/x", allowInternal=True)


if __name__ == "__main__":
    unittest.main()
