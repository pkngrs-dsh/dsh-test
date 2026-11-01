"""Bounded HTTP transfers with DNS-pinned connections for the legacy API.

Every connection, including redirects, resolves and validates its destination.
The socket connects to that exact address while HTTPS verifies the original
hostname. Environment proxy settings are deliberately not used.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import os
import socket
import ssl
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit


class TransportError(ValueError):
    """A public error that never contains a remote URL or response body."""


def _publicDnsOverride(host):
    """Optional explicit public routing; preserve HTTP Host and TLS hostname."""
    configured = os.environ.get("FACE_APPEARANCE_PUBLIC_DNS_OVERRIDES")
    if not configured:
        return None
    try:
        overrides = json.loads(configured)
        if not isinstance(overrides, dict):
            raise ValueError
        normalized = {}
        for name, value in overrides.items():
            if not isinstance(name, str) or not name:
                raise ValueError
            key = name.encode("idna").decode("ascii").lower().rstrip(".")
            values = [value] if isinstance(value, str) else value
            if not isinstance(values, list) or not values or key in normalized:
                raise ValueError
            addresses = []
            for item in values:
                if not isinstance(item, str) or "%" in item:
                    raise ValueError
                address = ipaddress.ip_address(item)
                effective = (address.ipv4_mapped
                             if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped
                             else address)
                if not effective.is_global or effective.is_multicast or effective.is_unspecified:
                    raise ValueError
                addresses.append(str(address))
            normalized[key] = list(dict.fromkeys(addresses))
        return normalized.get(host.lower().rstrip("."))
    except (ValueError, UnicodeError, TypeError):
        raise TransportError("公网 DNS 覆盖配置无效") from None


def resolveUrl(url: str, *, allowInternal: bool = False):
    if (not isinstance(url, str) or not url or len(url) > 8192
            or any(ord(char) <= 32 or ord(char) == 127 for char in url)):
        raise TransportError("图片或回调 URL 格式无效")
    try:
        url.encode("utf-8")  # Reject lone surrogate code points before connecting.
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError
        if parts.username is not None or parts.password is not None or parts.fragment:
            raise ValueError
        host = parts.hostname.encode("idna").decode("ascii")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if not 1 <= port <= 65535:
            raise ValueError
        overridden = _publicDnsOverride(host)
        addresses = [] if overridden is not None else socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except TransportError:
        raise
    except (ValueError, UnicodeError, OSError):
        raise TransportError("图片或回调 URL 无效或无法解析") from None
    ips = overridden if overridden is not None else list(dict.fromkeys(item[4][0] for item in addresses))
    if not ips:
        raise TransportError("图片或回调 URL 无法解析")
    for value in ips:
        address = ipaddress.ip_address(value)
        effective = address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped else address
        # Cloud metadata, multicast and unspecified addresses are never useful
        # image/callback targets, even during explicitly enabled internal use.
        if (effective.is_link_local or effective.is_multicast or effective.is_unspecified
                or ((overridden is not None or not allowInternal) and not effective.is_global)):
            raise TransportError("图片和回调 URL 必须指向公网 HTTP(S) 地址")
    return parts, host, port, ips


class _PinnedHttp(http.client.HTTPConnection):
    def __init__(self, host, port, address, *, timeout):
        super().__init__(host, port, timeout=timeout)
        self.address = address

    def connect(self):
        self.sock = socket.create_connection((self.address, self.port), self.timeout)


class _PinnedHttps(_PinnedHttp):
    default_port = 443

    def connect(self):
        raw = socket.create_connection((self.address, self.port), self.timeout)
        try:
            self.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


@contextmanager
def _request(url, *, method="GET", body=None, allowInternal=False, timeout=15):
    parts, host, port, addresses = resolveUrl(url, allowInternal=allowInternal)
    connectionClass = _PinnedHttps if parts.scheme == "https" else _PinnedHttp
    # Choosing a previously validated address closes the DNS-rebinding window.
    connection = None
    connectionDeadline = time.monotonic() + timeout
    for address in addresses:
        remaining = connectionDeadline - time.monotonic()
        if remaining <= 0:
            break
        candidate = connectionClass(host, port, address, timeout=remaining)
        try:
            candidate.connect()
            connection = candidate
            break
        except (OSError, http.client.HTTPException):
            candidate.close()
    if connection is None:
        raise TransportError("远端 HTTP 连接失败")
    # HTTP request targets must be ASCII. Preserve already escaped octets and
    # the URL delimiters, while encoding Unicode photo names exactly once.
    target = quote(parts.path or "/", safe="/%:@!$&'()*+,;=-._~")
    if parts.query:
        target += "?" + quote(parts.query, safe="/%?:@!$&'()*+,;=-._~")
    headers = {"User-Agent": "FaceAppearance-Legacy/1", "Accept-Encoding": "identity"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        connection.request(method, target, body=body, headers=headers)
        response = connection.getresponse()
        yield connection, response
    except TransportError:
        raise
    except (OSError, http.client.HTTPException, ValueError):
        raise TransportError("远端 HTTP 连接失败") from None
    finally:
        connection.close()


def downloadImage(url: str, destination: Path, *, maxBytes: int,
                  allowInternal: bool = False, timeout: float = 60.) -> int:
    deadline = time.monotonic() + timeout
    current = url
    for redirect in range(4):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TransportError("图片下载超时")
        with _request(current, allowInternal=allowInternal, timeout=min(15., remaining)) as (connection, response):
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if redirect == 3 or not location:
                    raise TransportError("图片重定向次数过多或地址缺失")
                current = urljoin(current, location)
                # The next iteration validates and pins the new destination.
                continue
            if response.status != 200:
                raise TransportError(f"图片下载失败（HTTP {response.status}）")
            if response.getheader("Content-Encoding", "identity").lower() not in {"", "identity"}:
                raise TransportError("图片服务器返回了不支持的压缩传输")
            length = response.getheader("Content-Length")
            if length is not None:
                try:
                    declared = int(length)
                except ValueError:
                    raise TransportError("图片长度声明无效") from None
                if declared < 1 or declared > maxBytes:
                    raise TransportError("图片为空或超过上传大小限制")
            received = 0
            with destination.open("xb") as stream:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TransportError("图片下载超时")
                    if connection.sock is not None:
                        connection.sock.settimeout(min(15., remaining))
                    chunk = response.read1(min(64 * 1024, maxBytes - received + 1))
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > maxBytes:
                        raise TransportError("图片超过上传大小限制")
                    stream.write(chunk)
            if received < 1 or (length is not None and received != declared):
                raise TransportError("图片下载不完整")
            return received
    raise TransportError("图片重定向次数过多")


def notifyResult(url: str, payload: dict, *, allowInternal: bool = False) -> int:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    with _request(url, method="POST", body=data, allowInternal=allowInternal, timeout=10.) as (_, response):
        # Do not forward callback payloads across redirects or read their body.
        if not 200 <= response.status < 300:
            raise TransportError(f"结果回调失败（HTTP {response.status}）")
        return response.status
