"""Dev worker tools: READ_FILE and LIST_FILES.

broker is not OS sandbox. No shell/write/search.
"""
from pathlib import Path
from dante.contracts import ToolManifest
from dante.tool_broker import ToolBroker, WorkspaceMethod, contained_path

_SENSITIVE_PARTS = {'.aws', '.azure', '.git', '.gnupg', '.ssh', '.env', '.npmrc',
                    '.pypirc', 'credentials', 'credentials.json', 'secrets.json',
                    'id_rsa', 'id_ed25519'}


def _sensitive(path: Path) -> bool:
    for part in (item.casefold() for item in path.parts):
        if part in _SENSITIVE_PARTS or 'credential' in part or 'secret' in part:
            return True
        if part.endswith(('.pem', '.key', '.p12', '.pfx')):
            return True
    return False


def _validate_path(root: Path, path: str) -> Path:
    if not isinstance(path, str):
        raise ValueError("path must be str")
    p = Path(path)
    if p.is_absolute():
        raise ValueError("absolute path rejected")
    if p.drive:
        raise ValueError("drive path rejected")
    if ".." in p.parts:
        raise ValueError("parent traversal rejected")
    resolved = contained_path(root, path)
    if _sensitive(Path(path)):
        raise ValueError('Sensitive paths are not available to the worker')
    return resolved


def _read_file(root: Path, path: str) -> dict:
    try:
        resolved = _validate_path(root, path)
    except ValueError as exc:
        if _sensitive(Path(path)):
            return {"ok": True, "path": path, "content": None,
                    "error": {"code": "blocked_path", "message": "This workspace path is not readable."}}
        raise
    # A readable tool response is still a successful tool operation when the
    # requested file is absent or unsuitable.  Raising here makes the broker
    # mark a harmless read as uncertain, which can strand the agent's required
    # evidence step and prevent an otherwise valid completion.
    def rejected(code: str, message: str) -> dict:
        return {"ok": True, "path": path, "content": None,
                "error": {"code": code, "message": message}}

    if not resolved.exists():
        return rejected("not_found", "No file exists at this workspace path.")
    if resolved.is_dir():
        return rejected("not_a_file", "The workspace path is a directory.")
    if not resolved.is_file() or resolved.is_symlink() or (hasattr(resolved, "is_junction") and resolved.is_junction()):
        return rejected("not_regular_file", "The workspace path is not a regular file.")
    try:
        if resolved.stat().st_size > 24 * 1024:
            return rejected("too_large", "The file exceeds the 24 KiB read limit.")
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return rejected("invalid_encoding", "The file is not valid UTF-8 text.")
    except OSError:
        return rejected("read_failed", "The workspace file could not be read.")
    return {"ok": True, "path": path, "content": content, "error": None}


def _list_files(root: Path, path: str) -> dict:
    resolved = _validate_path(root, path)
    if not resolved.is_dir() or resolved.is_symlink() or (hasattr(resolved, "is_junction") and resolved.is_junction()):
        raise ValueError("not a directory")
    entries = []
    truncated = False
    scanned = 0
    byte_count = 0
    for child in resolved.iterdir():
        scanned += 1
        if scanned > 1000:
            truncated = True
            break
        if (child.is_symlink() or (hasattr(child, "is_junction") and child.is_junction())
                or _sensitive(Path(child.name))):
            continue
        name = child.name
        if child.is_dir():
            entry_type = 'directory'
        elif child.is_file():
            entry_type = 'file'
        else:
            entry_type = 'other'
        size = len(name.encode("utf-8")) + len(entry_type) + 12
        if byte_count + size > 24000:
            truncated = True
            break
        entries.append({'name': name, 'type': entry_type})
        byte_count += size
        if len(entries) >= 100:
            truncated = True
            break
    entries.sort(key=lambda entry: entry['name'])
    return {"ok": True, "path": path, "entries": entries, "truncated": truncated}


def register_coding_tools(broker: ToolBroker, workspace: str) -> tuple:
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace must be a directory")
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    broker.register(
        ToolManifest(
            tool_id="READ_FILE",
            version="1",
            permissions=frozenset({"read_workspace"}),
            risk="R0",
            filesystem_scope=str(root),
            network_scope="none",
            secret_access="none",
            approval_policy="never",
            timeout_s=5,
            output_limit_bytes=32768,
            arguments_schema=schema,
            path_permissions={"path": "read_workspace"},
        ),
        WorkspaceMethod(
            handler=lambda path: _read_file(root, path),
            root=root,
            permission="read_workspace",
            schema=schema,
            paths={"path": "read_workspace"},
        ),
    )
    broker.register(
        ToolManifest(
            tool_id="LIST_FILES",
            version="1",
            permissions=frozenset({"read_workspace"}),
            risk="R0",
            filesystem_scope=str(root),
            network_scope="none",
            secret_access="none",
            approval_policy="never",
            timeout_s=5,
            output_limit_bytes=32768,
            arguments_schema=schema,
            path_permissions={"path": "read_workspace"},
        ),
        WorkspaceMethod(
            handler=lambda path: _list_files(root, path),
            root=root,
            permission="read_workspace",
            schema=schema,
            paths={"path": "read_workspace"},
        ),
    )
    return ("READ_FILE", "LIST_FILES")
