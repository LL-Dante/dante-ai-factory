"""
title: AI Cloud Safe Workspace
author: AI Cloud Free
version: 2.1.0
description: Strumenti filesystem e download confinati nella workspace AI Cloud.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import tempfile
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import httpx
from ddgs import DDGS


class ToolError(Exception):
    """Errore previsto e sicuro da mostrare al modello."""


class Tools:
    """Tool minimi per operare esclusivamente nella workspace autorizzata."""

    _MAX_TEXT_BYTES = 1_000_000
    _MAX_READ_CHARS = 200_000
    _MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024
    _MAX_DIRECTORY_ENTRIES = 500
    _RESERVED_NAMES = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    _MIME_EXTENSIONS = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "text/plain": ".txt",
        "text/markdown": ".md",
        "text/csv": ".csv",
        "application/json": ".json",
        "application/pdf": ".pdf",
    }

    def __init__(self) -> None:
        default_root = Path.home() / "Documents" / "AI-Cloud-Workspace"
        configured_root = os.environ.get("AI_CLOUD_WORKSPACE", str(default_root))
        self._root = Path(configured_root).expanduser().resolve()

        default_audit = (
            Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
            / "AI-Cloud-Free"
            / "logs"
            / "agent-tools.jsonl"
        )
        self._audit_path = Path(os.environ.get("AI_CLOUD_AGENT_AUDIT_LOG", str(default_audit)))
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_lock = threading.Lock()

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _model_name(metadata: dict | None) -> str:
        if not isinstance(metadata, dict):
            return "free-agent"
        model = metadata.get("model")
        if isinstance(model, dict):
            return str(model.get("id") or model.get("name") or "free-agent")
        return str(model or metadata.get("model_id") or "free-agent")

    def _audit(
        self,
        tool: str,
        status: str,
        started: float,
        metadata: dict | None = None,
        path: str | None = None,
        error_type: str | None = None,
    ) -> None:
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": self._model_name(metadata),
            "provider": "free-agent-zero-cost-route",
            "tool": tool,
            "status": status,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "path": path,
            "error_type": error_type,
        }
        line = self._json(event)
        with self._audit_lock:
            with self._audit_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        if isinstance(exc, ToolError):
            return str(exc)
        if isinstance(exc, OSError):
            details = []
            if exc.errno is not None:
                details.append(f"errno={exc.errno}")
            winerror = getattr(exc, "winerror", None)
            if winerror is not None:
                details.append(f"winerror={winerror}")
            code = f" ({', '.join(details)})" if details else ""
            message = exc.strerror or type(exc).__name__
            return f"Errore filesystem: {message}{code}."
        return f"Errore tecnico ({type(exc).__name__})."

    @classmethod
    def _sanitize_component(cls, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        normalized = re.sub(r"\s+", "_", normalized.strip())
        normalized = re.sub(r"[^A-Za-z0-9._()\-]", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip(" ._")
        if not normalized:
            raise ToolError("Il nome del file o della cartella non e valido.")
        if normalized.split(".", 1)[0].upper() in cls._RESERVED_NAMES:
            raise ToolError("Nome riservato da Windows non consentito.")
        return normalized[:120]

    def _safe_path(self, relative_path: str, *, allow_root: bool = False) -> Path:
        raw = str(relative_path or "").strip()
        if allow_root and raw in {"", "."}:
            return self._root
        if not raw or len(raw) > 240:
            raise ToolError("Percorso relativo vuoto o troppo lungo.")

        normalized = raw.replace("\\", "/")
        if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
            raise ToolError("Sono consentiti soltanto percorsi relativi alla workspace.")

        raw_parts = [part for part in normalized.split("/") if part]
        if not raw_parts or any(part in {".", ".."} for part in raw_parts):
            raise ToolError("Path traversal o percorso non valido bloccato.")

        safe_parts = [self._sanitize_component(part) for part in raw_parts]
        candidate = self._root.joinpath(*safe_parts).resolve(strict=False)
        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            raise ToolError("Accesso fuori dalla workspace bloccato.") from exc
        return candidate

    def _relative(self, path: Path) -> str:
        return str(path.resolve(strict=False).relative_to(self._root)).replace("\\", "/") or "."

    def _atomic_write(self, target: Path, data: bytes, overwrite: bool) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = target.parent.resolve()
        try:
            resolved_parent.relative_to(self._root)
        except ValueError as exc:
            raise ToolError("La cartella di destinazione esce dalla workspace.") from exc
        if target.exists() and not overwrite:
            raise ToolError("Il file esiste gia; specificare overwrite=true per sostituirlo.")

        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=resolved_parent, prefix=".ai-cloud-", suffix=".tmp", delete=False) as handle:
                temp_name = handle.name
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    @staticmethod
    def _validate_public_url(url: str) -> str:
        parsed = urlparse(str(url).strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ToolError("Sono consentiti soltanto URL pubblici HTTP o HTTPS.")
        if parsed.username or parsed.password:
            raise ToolError("URL con credenziali incorporate non consentiti.")
        try:
            addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        except socket.gaierror as exc:
            raise ToolError("Impossibile risolvere il nome host.") from exc
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                raise ToolError("Download da indirizzi locali o riservati bloccato.")
        return parsed.geturl()

    @classmethod
    def _verify_content(cls, content_type: str, data: bytes) -> None:
        if content_type == "image/jpeg" and not data.startswith(b"\xff\xd8\xff"):
            raise ToolError("Il contenuto non corrisponde a un'immagine JPEG valida.")
        if content_type == "image/png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ToolError("Il contenuto non corrisponde a un'immagine PNG valida.")
        if content_type == "image/gif" and not data.startswith((b"GIF87a", b"GIF89a")):
            raise ToolError("Il contenuto non corrisponde a un'immagine GIF valida.")
        if content_type == "image/webp" and not (data.startswith(b"RIFF") and data[8:12] == b"WEBP"):
            raise ToolError("Il contenuto non corrisponde a un'immagine WebP valida.")
        if content_type == "application/pdf" and not data.startswith(b"%PDF-"):
            raise ToolError("Il contenuto non corrisponde a un PDF valido.")
        if content_type.startswith("text/") or content_type == "application/json":
            if b"\x00" in data[:4096]:
                raise ToolError("Payload binario sospetto rifiutato.")

    @classmethod
    def _filename_from_response(cls, response: httpx.Response, url: str) -> str:
        disposition = response.headers.get("content-disposition", "")
        match = re.search(r"filename\*=UTF-8''([^;]+)", disposition, flags=re.IGNORECASE)
        if not match:
            match = re.search(r'filename="?([^";]+)', disposition, flags=re.IGNORECASE)
        value = unquote(match.group(1)) if match else unquote(Path(urlparse(url).path).name)
        return cls._sanitize_component(value or "download")

    def ensure_workspace(self, __metadata__: dict = None) -> str:
        """Create and verify the configured workspace with a real write/read probe.

        :return: JSON with the absolute workspace path and verified read/write flags.
        """
        started = time.perf_counter()
        probe: Path | None = None
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            if not self._root.is_dir():
                raise ToolError("Il percorso workspace non e una cartella.")
            probe = self._root / f".ai-cloud-write-probe-{os.getpid()}-{threading.get_ident()}.tmp"
            probe.write_bytes(b"ai-cloud-workspace-ok")
            readable = probe.read_bytes() == b"ai-cloud-workspace-ok"
            probe.unlink()
            probe = None
            if not readable:
                raise ToolError("La verifica di lettura della workspace non e riuscita.")
            result = {
                "ok": True,
                "workspace": str(self._root),
                "exists": True,
                "readable": True,
                "writable": True,
            }
            self._audit("ensure_workspace", "ok", started, __metadata__, str(self._root))
            return self._json(result)
        except Exception as exc:
            if probe and probe.exists():
                try:
                    probe.unlink()
                except OSError:
                    pass
            self._audit("ensure_workspace", "error", started, __metadata__, str(self._root), type(exc).__name__)
            return self._json(
                {
                    "ok": False,
                    "workspace": str(self._root),
                    "exists": self._root.is_dir(),
                    "readable": os.access(self._root, os.R_OK),
                    "writable": False,
                    "error": self._safe_error(exc),
                }
            )

    def create_directory(self, relative_path: str, __metadata__: dict = None) -> str:
        """Create a directory inside the safe workspace.

        :param relative_path: Descriptive path relative to the workspace root.
        :return: JSON containing the created directory path.
        """
        started = time.perf_counter()
        path: Path | None = None
        try:
            path = self._safe_path(relative_path)
            path.mkdir(parents=True, exist_ok=True)
            result = {"ok": True, "path": str(path), "relative_path": self._relative(path)}
            self._audit("create_directory", "ok", started, __metadata__, str(path))
            return self._json(result)
        except Exception as exc:
            self._audit("create_directory", "error", started, __metadata__, str(path) if path else None, type(exc).__name__)
            return self._json({"ok": False, "error": self._safe_error(exc)})

    def list_directory(self, relative_path: str = "", __metadata__: dict = None) -> str:
        """List files and folders in one workspace directory without recursion.

        :param relative_path: Directory relative to the workspace root; leave empty for the root.
        :return: JSON list with names, types, sizes, and relative paths.
        """
        started = time.perf_counter()
        path: Path | None = None
        try:
            path = self._safe_path(relative_path, allow_root=True)
            if not path.is_dir():
                raise ToolError("La cartella richiesta non esiste.")
            entries = []
            for item in sorted(path.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower())):
                if len(entries) >= self._MAX_DIRECTORY_ENTRIES:
                    break
                stat = item.stat()
                entries.append(
                    {
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size_bytes": None if item.is_dir() else stat.st_size,
                        "relative_path": self._relative(item),
                    }
                )
            self._audit("list_directory", "ok", started, __metadata__, str(path))
            return self._json({"ok": True, "path": str(path), "entries": entries, "truncated": len(entries) >= self._MAX_DIRECTORY_ENTRIES})
        except Exception as exc:
            self._audit("list_directory", "error", started, __metadata__, str(path) if path else None, type(exc).__name__)
            return self._json({"ok": False, "error": self._safe_error(exc)})

    def read_file(self, relative_path: str, max_chars: int = 100000, __metadata__: dict = None) -> str:
        """Read a UTF-8 text file from the safe workspace.

        :param relative_path: File path relative to the workspace root.
        :param max_chars: Maximum characters returned, capped at 200000.
        :return: JSON with file content and truncation status.
        """
        started = time.perf_counter()
        path: Path | None = None
        try:
            path = self._safe_path(relative_path)
            if not path.is_file():
                raise ToolError("Il file richiesto non esiste.")
            if path.stat().st_size > self._MAX_TEXT_BYTES:
                raise ToolError("File troppo grande per la lettura controllata.")
            data = path.read_bytes()
            if b"\x00" in data[:4096]:
                raise ToolError("La lettura diretta di file binari non e consentita.")
            text = data.decode("utf-8-sig")
            limit = max(1, min(int(max_chars), self._MAX_READ_CHARS))
            truncated = len(text) > limit
            self._audit("read_file", "ok", started, __metadata__, str(path))
            return self._json({"ok": True, "path": str(path), "content": text[:limit], "truncated": truncated})
        except Exception as exc:
            self._audit("read_file", "error", started, __metadata__, str(path) if path else None, type(exc).__name__)
            return self._json({"ok": False, "error": self._safe_error(exc)})

    def write_file(self, relative_path: str, content: str, overwrite: bool = True, __metadata__: dict = None) -> str:
        """Write UTF-8 text inside the safe workspace using an atomic replace.

        :param relative_path: Destination file path relative to the workspace root.
        :param content: UTF-8 text to write.
        :param overwrite: Whether an existing file may be replaced.
        :return: JSON containing the final path and byte count.
        """
        started = time.perf_counter()
        path: Path | None = None
        try:
            path = self._safe_path(relative_path)
            data = str(content).encode("utf-8")
            if len(data) > self._MAX_TEXT_BYTES:
                raise ToolError("Contenuto troppo grande: limite 1 MB.")
            self._atomic_write(path, data, bool(overwrite))
            self._audit("write_file", "ok", started, __metadata__, str(path))
            return self._json({"ok": True, "path": str(path), "relative_path": self._relative(path), "size_bytes": len(data)})
        except Exception as exc:
            self._audit("write_file", "error", started, __metadata__, str(path) if path else None, type(exc).__name__)
            return self._json({"ok": False, "error": self._safe_error(exc)})

    def write_text_file(self, relative_path: str, content: str, __metadata__: dict = None) -> str:
        """Write or replace one UTF-8 text file inside the workspace.

        :param relative_path: Destination path relative to the workspace root.
        :param content: UTF-8 text to save.
        :return: JSON with the unambiguous absolute and relative paths.
        """
        started = time.perf_counter()
        path: Path | None = None
        try:
            path = self._safe_path(relative_path)
            data = str(content).encode("utf-8")
            if len(data) > self._MAX_TEXT_BYTES:
                raise ToolError("Contenuto troppo grande: limite 1 MB.")
            self._atomic_write(path, data, True)
            result = {
                "ok": True,
                "workspace": str(self._root),
                "path": str(path),
                "relative_path": self._relative(path),
                "size_bytes": len(data),
            }
            self._audit("write_text_file", "ok", started, __metadata__, str(path))
            return self._json(result)
        except Exception as exc:
            self._audit("write_text_file", "error", started, __metadata__, str(path) if path else None, type(exc).__name__)
            return self._json({"ok": False, "workspace": str(self._root), "error": self._safe_error(exc)})

    def file_exists(self, relative_path: str, __metadata__: dict = None) -> str:
        """Check whether a relative workspace path exists.

        :param relative_path: File or directory path relative to the workspace.
        :return: JSON with exists, type, absolute path, and relative path.
        """
        started = time.perf_counter()
        path: Path | None = None
        try:
            path = self._safe_path(relative_path)
            exists = path.exists()
            kind = "directory" if path.is_dir() else "file" if path.is_file() else None
            result = {
                "ok": True,
                "workspace": str(self._root),
                "exists": exists,
                "type": kind,
                "path": str(path),
                "relative_path": self._relative(path),
            }
            self._audit("file_exists", "ok", started, __metadata__, str(path))
            return self._json(result)
        except Exception as exc:
            self._audit("file_exists", "error", started, __metadata__, str(path) if path else None, type(exc).__name__)
            return self._json({"ok": False, "workspace": str(self._root), "error": self._safe_error(exc)})

    def save_text(self, relative_path: str, text: str, overwrite: bool = True, __metadata__: dict = None) -> str:
        """Save plain text or Markdown in the workspace.

        :param relative_path: Destination .txt or .md path relative to the workspace.
        :param text: Text to save.
        :param overwrite: Whether an existing file may be replaced.
        :return: JSON containing the final path.
        """
        started = time.perf_counter()
        path: Path | None = None
        try:
            path = self._safe_path(relative_path)
            if path.suffix.lower() not in {".txt", ".md"}:
                path = path.with_suffix(".txt")
            data = str(text).encode("utf-8")
            if len(data) > self._MAX_TEXT_BYTES:
                raise ToolError("Contenuto troppo grande: limite 1 MB.")
            self._atomic_write(path, data, bool(overwrite))
            self._audit("save_text", "ok", started, __metadata__, str(path))
            return self._json({"ok": True, "path": str(path), "relative_path": self._relative(path), "size_bytes": len(data)})
        except Exception as exc:
            self._audit("save_text", "error", started, __metadata__, str(path) if path else None, type(exc).__name__)
            return self._json({"ok": False, "error": self._safe_error(exc)})

    def save_json(self, relative_path: str, data: dict, overwrite: bool = True, __metadata__: dict = None) -> str:
        """Serialize a JSON object safely inside the workspace.

        :param relative_path: Destination path relative to the workspace.
        :param data: JSON object to serialize.
        :param overwrite: Whether an existing file may be replaced.
        :return: JSON containing the final path.
        """
        started = time.perf_counter()
        path: Path | None = None
        try:
            path = self._safe_path(relative_path)
            if path.suffix.lower() != ".json":
                path = path.with_suffix(".json")
            payload = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            if len(payload) > self._MAX_TEXT_BYTES:
                raise ToolError("JSON troppo grande: limite 1 MB.")
            self._atomic_write(path, payload, bool(overwrite))
            self._audit("save_json", "ok", started, __metadata__, str(path))
            return self._json({"ok": True, "path": str(path), "relative_path": self._relative(path), "size_bytes": len(payload)})
        except Exception as exc:
            self._audit("save_json", "error", started, __metadata__, str(path) if path else None, type(exc).__name__)
            return self._json({"ok": False, "error": self._safe_error(exc)})

    def search_images(self, query: str, count: int = 5, __metadata__: dict = None) -> str:
        """Search public image results with the free DuckDuckGo backend.

        :param query: Descriptive image search query.
        :param count: Number of results, capped at 8.
        :return: JSON with direct image URLs and their source pages.
        """
        started = time.perf_counter()
        try:
            limit = max(1, min(int(count), 8))
            with DDGS() as ddgs:
                raw_results = list(ddgs.images(str(query), safesearch="moderate", max_results=limit) or [])
            results = []
            for item in raw_results:
                image_url = str(item.get("image") or "")
                if urlparse(image_url).scheme not in {"http", "https"}:
                    continue
                results.append(
                    {
                        "title": item.get("title"),
                        "image_url": image_url,
                        "source_page": item.get("url"),
                        "thumbnail": item.get("thumbnail"),
                    }
                )
            self._audit("search_images", "ok", started, __metadata__)
            return self._json({"ok": True, "results": results[:limit]})
        except Exception as exc:
            self._audit("search_images", "error", started, __metadata__, error_type=type(exc).__name__)
            return self._json({"ok": False, "error": self._safe_error(exc)})

    def download_file(self, url: str, relative_path: str = "", overwrite: bool = False, __metadata__: dict = None) -> str:
        """Download a small public image, text, JSON, CSV, Markdown, or PDF into the workspace.

        The tool rejects private-network destinations, unsafe MIME types, oversized files,
        suspicious payloads, path traversal, and more than five redirects. Files are never executed.

        :param url: Public HTTP or HTTPS URL.
        :param relative_path: Optional destination file or directory relative to the workspace.
        :param overwrite: Whether an existing file may be replaced.
        :return: JSON containing the verified MIME type, byte count, and final local path.
        """
        started = time.perf_counter()
        target: Path | None = None
        temp_name: str | None = None
        try:
            current_url = self._validate_public_url(url)
            headers = {"User-Agent": "AI-Cloud-Free-Agent/2.0 (+local-safe-downloader)"}
            with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0), headers=headers, follow_redirects=False) as client:
                for _ in range(6):
                    self._validate_public_url(current_url)
                    with client.stream("GET", current_url) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location:
                                raise ToolError("Redirect privo di destinazione.")
                            current_url = self._validate_public_url(urljoin(current_url, location))
                            continue
                        if response.status_code < 200 or response.status_code >= 300:
                            raise ToolError(f"Download non riuscito: HTTP {response.status_code}.")

                        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                        if content_type == "image/jpg":
                            content_type = "image/jpeg"
                        if content_type not in self._MIME_EXTENSIONS:
                            raise ToolError(f"Content-Type non consentito: {content_type or 'mancante'}.")

                        declared = response.headers.get("content-length")
                        if declared and int(declared) > self._MAX_DOWNLOAD_BYTES:
                            raise ToolError("File troppo grande: limite 10 MB.")

                        remote_name = self._filename_from_response(response, current_url)
                        if relative_path:
                            requested = self._safe_path(relative_path)
                            target = requested / remote_name if not requested.suffix else requested
                        else:
                            target = self._safe_path(remote_name)

                        expected_extension = self._MIME_EXTENSIONS[content_type]
                        valid_extensions = {expected_extension}
                        if content_type == "image/jpeg":
                            valid_extensions.add(".jpeg")
                        if target.suffix.lower() not in valid_extensions:
                            target = target.with_suffix(expected_extension)

                        target.parent.mkdir(parents=True, exist_ok=True)
                        resolved_parent = target.parent.resolve()
                        try:
                            resolved_parent.relative_to(self._root)
                        except ValueError as exc:
                            raise ToolError("Destinazione fuori dalla workspace bloccata.") from exc
                        if target.exists() and not overwrite:
                            raise ToolError("Il file esiste gia; usare un nuovo nome o overwrite=true.")

                        total = 0
                        with tempfile.NamedTemporaryFile(dir=resolved_parent, prefix=".download-", suffix=".part", delete=False) as handle:
                            temp_name = handle.name
                            for chunk in response.iter_bytes(64 * 1024):
                                total += len(chunk)
                                if total > self._MAX_DOWNLOAD_BYTES:
                                    raise ToolError("File troppo grande: limite 10 MB.")
                                handle.write(chunk)
                            handle.flush()
                            os.fsync(handle.fileno())

                        data = Path(temp_name).read_bytes()
                        if not data:
                            raise ToolError("Il server ha restituito un file vuoto.")
                        self._verify_content(content_type, data)
                        os.replace(temp_name, target)
                        temp_name = None
                        result = {
                            "ok": True,
                            "path": str(target),
                            "relative_path": self._relative(target),
                            "content_type": content_type,
                            "size_bytes": total,
                            "source_url": current_url,
                            "executed": False,
                        }
                        self._audit("download_file", "ok", started, __metadata__, str(target))
                        return self._json(result)
                raise ToolError("Troppi redirect durante il download.")
        except Exception as exc:
            self._audit("download_file", "error", started, __metadata__, str(target) if target else None, type(exc).__name__)
            return self._json({"ok": False, "error": self._safe_error(exc)})
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    def download_image(self, url: str, relative_path: str, __metadata__: dict = None) -> str:
        """Download and verify one public JPEG, PNG, GIF, or WebP image.

        Non-image content, HTML, oversized payloads, private addresses, and path traversal are rejected.

        :param url: Direct public HTTP or HTTPS image URL.
        :param relative_path: Destination path relative to the workspace.
        :return: JSON with verified image MIME, size, and final absolute path.
        """
        started = time.perf_counter()
        result = json.loads(self.download_file(url, relative_path, False, __metadata__))
        if not result.get("ok"):
            self._audit("download_image", "error", started, __metadata__, error_type="DownloadRejected")
            return self._json(result)
        content_type = str(result.get("content_type") or "")
        if not content_type.startswith("image/"):
            saved = Path(str(result.get("path") or "")).resolve(strict=False)
            try:
                saved.relative_to(self._root)
                if saved.is_file():
                    saved.unlink()
            except (OSError, ValueError):
                pass
            error = "Il contenuto scaricato non e un'immagine consentita."
            self._audit("download_image", "error", started, __metadata__, str(saved), "InvalidImageContentType")
            return self._json({"ok": False, "workspace": str(self._root), "error": error})
        self._audit("download_image", "ok", started, __metadata__, str(result.get("path")))
        return self._json(result)
