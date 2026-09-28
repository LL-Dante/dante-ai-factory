"""Bounded workspace search and write tools for the development worker.

ToolBroker enforces application-level permissions and paths; it is not an OS sandbox.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from dante.contracts import ToolManifest
from dante.tool_broker import ToolBroker, WorkspaceMethod, contained_path

_MAX_FILE_BYTES = 24 * 1024
_MAX_EDIT_FILE_BYTES = 4 * 1024 * 1024
_MAX_RANGE_LINES = 100
_MAX_RANGE_BYTES = 8 * 1024
_MAX_PATCH_BYTES = 8 * 1024
_MAX_RESULT_BYTES = 24 * 1024
_SENSITIVE_PARTS = {'.aws', '.azure', '.git', '.gnupg', '.ssh', '.env', '.npmrc',
                    '.pypirc', 'credentials', 'credentials.json', 'secrets.json',
                    'id_rsa', 'id_ed25519'}


def _sensitive(path: Path) -> bool:
    return any(part.casefold() in _SENSITIVE_PARTS or 'credential' in part.casefold()
               or 'secret' in part.casefold() or part.casefold().endswith(('.pem', '.key', '.p12', '.pfx'))
               for part in path.parts)


def _resolve(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ValueError('Invalid relative path')
    candidate = Path(value)
    if candidate.is_absolute() or candidate.drive or '..' in candidate.parts:
        raise ValueError('Path must stay relative to the workspace')
    if _sensitive(candidate):
        raise ValueError('Sensitive paths are not available to the worker')
    return contained_path(root, value)


def _is_reparse(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction())


def _structured_error(code: str, message: str, **details) -> dict:
    return {'ok': True, 'operation_ok': False,
            'error': {'code': code, 'message': message, **details}}


def _search(root: Path, path: str, query: str) -> dict:
    if not isinstance(query, str) or not 1 <= len(query) <= 200:
        raise ValueError('Query must contain 1 to 200 characters')
    start = _resolve(root, path)
    if _is_reparse(start) or not start.exists():
        raise ValueError('Search path is unavailable')
    files = []
    truncated = False
    if start.is_file():
        files = [start]
    elif start.is_dir():
        visited_dirs = 0
        for current, dirs, names in os.walk(start, topdown=True, followlinks=False):
            current_path = Path(current)
            depth = len(current_path.relative_to(start).parts)
            visited_dirs += 1
            dirs[:] = [name for name in dirs if depth < 4
                       and not _is_reparse(current_path / name)
                       and not _sensitive(Path(name))]
            if visited_dirs > 250:
                truncated = True
                break
            for name in names:
                candidate = current_path / name
                if _is_reparse(candidate) or _sensitive(Path(name)):
                    continue
                files.append(candidate)
                if len(files) >= 250:
                    truncated = True
                    break
            if truncated:
                break
    else:
        raise ValueError('Search target must be a file or directory')

    matches = []
    used_bytes = 0
    needle = query.casefold()
    for candidate in files:
        try:
            if _is_reparse(candidate) or not candidate.is_file() or candidate.stat().st_size > _MAX_FILE_BYTES:
                continue
            raw = candidate.read_bytes()
            if len(raw) > _MAX_FILE_BYTES:
                continue
            content = raw.decode('utf-8')
        except (OSError, UnicodeError):
            continue
        for line_number, line in enumerate(content.splitlines(), 1):
            if needle not in line.casefold():
                continue
            relative = candidate.relative_to(root).as_posix()
            item = {'path': relative, 'line': line_number, 'text': line[:2000]}
            item_bytes = len(str(item).encode('utf-8'))
            if len(matches) >= 100 or used_bytes + item_bytes > _MAX_RESULT_BYTES:
                truncated = True
                break
            matches.append(item)
            used_bytes += item_bytes
        if truncated:
            break
    return {'ok': True, 'matches': matches, 'truncated': truncated}


def _write(root: Path, path: str, content: str) -> dict:
    if not isinstance(content, str):
        raise ValueError('Content must be text')
    try:
        data = content.encode('utf-8')
    except UnicodeError:
        raise ValueError('Content must be valid UTF-8 text') from None
    if len(data) > _MAX_FILE_BYTES:
        raise ValueError('Content exceeds the 24 KiB limit')
    target = _resolve(root, path)
    if _is_reparse(target) or (target.exists() and target.is_dir()):
        raise ValueError('Target is a link or directory')
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    _resolve(root, path)
    if _is_reparse(target) or (target.exists() and target.is_dir()):
        raise ValueError('Target changed during write preparation')
    fd, temporary = tempfile.mkstemp(prefix='.dante-', suffix='.tmp', dir=parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return {'ok': True, 'path': path, 'bytes_written': len(data)}


def _read_range(root: Path, path: str, start_line: int, end_line: int) -> dict:
    if (type(start_line) is not int or type(end_line) is not int or start_line < 1
            or end_line < start_line or end_line - start_line + 1 > _MAX_RANGE_LINES):
        return _structured_error('invalid_line_range', 'Use a 1-based range of at most 100 lines.')
    target = _resolve(root, path)
    if _is_reparse(target) or not target.exists() or not target.is_file():
        return _structured_error('file_unavailable', 'Target file is unavailable.')
    try:
        if target.stat().st_size > _MAX_EDIT_FILE_BYTES:
            return _structured_error('file_too_large', 'Target exceeds the 4 MiB read limit.')
        content = target.read_bytes().decode('utf-8')
    except UnicodeError:
        return _structured_error('invalid_encoding', 'Target is not UTF-8 text.')
    except OSError:
        return _structured_error('file_unavailable', 'Target file is unavailable.')
    lines = content.splitlines()
    selected = []
    used = 0
    for number in range(start_line, min(end_line, len(lines)) + 1):
        line = lines[number - 1]
        encoded_size = len(line.encode('utf-8'))
        if used + encoded_size > _MAX_RANGE_BYTES:
            break
        selected.append({'line': number, 'text': line})
        used += encoded_size
    return {'ok': True, 'path': target.relative_to(root).as_posix(),
            'start_line': start_line, 'end_line': end_line, 'total_lines': len(lines),
            'returned_lines': len(selected), 'truncated': len(selected) < max(0, min(end_line, len(lines)) - start_line + 1),
            'result_sha256': hashlib.sha256(str(selected).encode('utf-8')).hexdigest(),
            'lines': selected}


def _patch(root: Path, path: str, expected: str, replacement: str) -> dict:
    if not isinstance(expected, str) or not expected:
        return _structured_error('invalid_expected_text', 'Expected text must be non-empty.')
    if not isinstance(replacement, str):
        return _structured_error('invalid_replacement', 'Replacement must be text.')
    try:
        patch_bytes = len(expected.encode('utf-8')) + len(replacement.encode('utf-8'))
    except UnicodeError:
        return _structured_error('invalid_encoding', 'Patch must be valid UTF-8 text.')
    if patch_bytes > _MAX_PATCH_BYTES:
        return _structured_error('patch_too_large', 'Patch exceeds the 8 KiB limit.')
    target = _resolve(root, path)
    if _is_reparse(target) or not target.exists() or not target.is_file():
        return _structured_error('file_unavailable', 'Target file is unavailable.')
    try:
        if target.stat().st_size > _MAX_EDIT_FILE_BYTES:
            return _structured_error('file_too_large', 'Target exceeds the 4 MiB edit limit.')
        original_bytes = target.read_bytes()
        original = original_bytes.decode('utf-8')
    except UnicodeError:
        return _structured_error('invalid_encoding', 'Target is not UTF-8 text.')
    except OSError:
        return _structured_error('file_unavailable', 'Target file is unavailable.')
    occurrences = original.count(expected)
    if occurrences != 1:
        return _structured_error('expected_text_mismatch' if occurrences == 0 else 'expected_text_ambiguous',
                                 'Expected text must match exactly once.', match_count=occurrences)
    updated = original.replace(expected, replacement, 1)
    updated_bytes = updated.encode('utf-8')
    before_hash = hashlib.sha256(original_bytes).hexdigest()
    after_hash = hashlib.sha256(updated_bytes).hexdigest()
    try:
        if hashlib.sha256(target.read_bytes()).hexdigest() != before_hash:
            return _structured_error('file_changed', 'Target changed before patch could be applied.')
        mode = target.stat().st_mode
        fd, temporary = tempfile.mkstemp(prefix='.dante-patch-', suffix='.tmp', dir=target.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(updated_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, mode)
            _resolve(root, path)
            if _is_reparse(target) or hashlib.sha256(target.read_bytes()).hexdigest() != before_hash:
                os.unlink(temporary)
                return _structured_error('file_changed', 'Target changed before patch could be applied.')
            os.replace(temporary, target)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    except OSError:
        return _structured_error('patch_failed', 'Patch could not be applied.')
    first_line = original.count('\n', 0, original.find(expected)) + 1
    return {'ok': True, 'operation_ok': True, 'path': target.relative_to(root).as_posix(), 'pre_sha256': before_hash,
            'post_sha256': after_hash, 'changed_region': {'start_line': first_line,
            'end_line': first_line + replacement.count('\n')}}


def register_edit_tools(broker: ToolBroker, workspace: str) -> tuple[str, ...]:
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Workspace must be an existing directory')
    specifications = (
        ('SEARCH_TEXT', 'read_workspace', 'R0',
         {'type': 'object', 'properties': {'path': {'type': 'string'}, 'query': {'type': 'string'}},
          'required': ['path', 'query'], 'additionalProperties': False},
         lambda path, query: _search(root, path, query)),
        ('READ_FILE_RANGE', 'read_workspace', 'R0',
         {'type': 'object', 'properties': {'path': {'type': 'string'}, 'start_line': {'type': 'integer'}, 'end_line': {'type': 'integer'}},
          'required': ['path', 'start_line', 'end_line'], 'additionalProperties': False},
         lambda path, start_line, end_line: _read_range(root, path, start_line, end_line)),
        ('PATCH_FILE', 'write_workspace', 'R1',
         {'type': 'object', 'properties': {'path': {'type': 'string'}, 'expected': {'type': 'string'}, 'replacement': {'type': 'string'}},
          'required': ['path', 'expected', 'replacement'], 'additionalProperties': False},
         lambda path, expected, replacement: _patch(root, path, expected, replacement)),
        ('WRITE_FILE', 'write_workspace', 'R1',
         {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
          'required': ['path', 'content'], 'additionalProperties': False},
         lambda path, content: _write(root, path, content)),
    )
    for name, permission, risk, schema, handler in specifications:
        paths = {'path': permission}
        manifest = ToolManifest(
            tool_id=name, version='1', permissions=frozenset({permission}), risk=risk,
            filesystem_scope=str(root), network_scope='none', secret_access='none',
            approval_policy='never', timeout_s=5, output_limit_bytes=32768,
            arguments_schema=schema, path_permissions=paths,
        )
        broker.register(manifest, WorkspaceMethod(handler=handler, root=root,
                                                    permission=permission, schema=schema, paths=paths))
    return ('SEARCH_TEXT', 'READ_FILE_RANGE', 'PATCH_FILE', 'WRITE_FILE')
