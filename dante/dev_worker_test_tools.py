"""Fixed, bounded stdlib unittest runner for development workspaces.

This provides process cleanup and resource limits, not an OS security sandbox.
Tests execute as the current user with that user's filesystem/network access.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path, PureWindowsPath

from dante.contracts import ToolManifest
from dante.tool_broker import ToolBroker, WorkspaceMethod, contained_path

_TIMEOUT_S = 120
_OUTPUT_LIMIT = 32 * 1024
_TEST_FILE = re.compile(r'test_[A-Za-z0-9_]+\.py\Z')


def _run_tests(root: Path, target: str) -> dict:
    _validate_target(root, target)
    normalized = target.replace('\\', '/')
    parts = normalized.split('/')

    env = {'PATH': os.environ.get('PATH', ''), 'SYSTEMROOT': os.environ.get('SYSTEMROOT', ''),
           'WINDIR': os.environ.get('WINDIR', ''), 'TEMP': os.environ.get('TEMP', ''),
           'TMP': os.environ.get('TMP', ''), 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1',
           'PYTHONDONTWRITEBYTECODE': '1'}
    creationflags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
    started = time.monotonic()
    module_path = (root / 'tests' / parts[1]).as_posix()
    bootstrap = (
        'import importlib.util,sys,unittest; '
        'spec=importlib.util.spec_from_file_location("_dante_selected_test", ' + repr(module_path) + '); '
        'module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); '
        'suite=unittest.defaultTestLoader.loadTestsFromModule(module); '
        'result=unittest.TextTestRunner(verbosity=2).run(suite); '
        'sys.exit(0 if result.wasSuccessful() else 1)'
    )
    process = subprocess.Popen(
        [sys.executable, '-I', '-c', bootstrap],
        cwd=root, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False,
        creationflags=creationflags,
    )
    output = bytearray()
    overflow = threading.Event()

    def drain() -> None:
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                return
            remaining = _OUTPUT_LIMIT - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                overflow.set()
                if process.poll() is None:
                    process.kill()
                return

    reader = threading.Thread(target=drain, name='dante-test-output', daemon=True)
    reader.start()
    timed_out = False
    try:
        process.wait(timeout=_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == 'nt':
            # Targets only this process tree; no unrelated processes are addressed.
            subprocess.run(['taskkill.exe', '/PID', str(process.pid), '/T', '/F'],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10, check=False)
        else:
            process.kill()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
    reader.join(timeout=5)
    text = bytes(output).decode('utf-8', errors='replace')
    return {'ok': process.returncode == 0 and not timed_out and not overflow.is_set(),
            'target': normalized, 'return_code': process.returncode,
            'timed_out': timed_out, 'output_truncated': overflow.is_set(),
            'duration_s': round(time.monotonic() - started, 3), 'output': text,
            'isolation': 'bounded subprocess; current-user permissions; not an OS sandbox'}


def _validate_target(root: Path, target: str) -> Path:
    if not isinstance(target, str) or not target or len(target) > 256:
        raise ValueError('Target must be a relative tests/test_*.py path')
    win = PureWindowsPath(target)
    path = Path(target)
    if path.is_absolute() or win.is_absolute() or win.drive or '..' in win.parts or '\x00' in target:
        raise ValueError('Target must be a relative tests/test_*.py path')
    normalized = target.replace('\\', '/')
    parts = normalized.split('/')
    if len(parts) != 2 or parts[0] != 'tests' or not _TEST_FILE.fullmatch(parts[1]):
        raise ValueError('Only one tests/test_*.py file may run')
    candidate = contained_path(root, normalized)
    tests_dir = root / 'tests'
    if (candidate.parent != tests_dir or candidate.is_symlink()
            or (hasattr(candidate, 'is_junction') and candidate.is_junction())
            or not candidate.is_file()):
        raise ValueError('Test target is not a regular file in workspace tests/')
    return candidate


def register_test_tool(broker: ToolBroker, workspace: str) -> tuple[str, ...]:
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Workspace must exist')
    schema = {'type': 'object', 'properties': {'target': {'type': 'string'}},
              'required': ['target'], 'additionalProperties': False}
    broker.register(ToolManifest(
        tool_id='RUN_TESTS', version='1', permissions=frozenset({'read_workspace'}),
        risk='R1', filesystem_scope=str(root), network_scope='none', secret_access='none',
        approval_policy='never', timeout_s=_TIMEOUT_S + 20, output_limit_bytes=_OUTPUT_LIMIT + 4096,
        arguments_schema=schema, path_permissions={'target': 'read_workspace'}),
        WorkspaceMethod(handler=lambda target: _run_tests(root, target), root=root,
                        permission='read_workspace', schema=schema,
                        paths={'target': 'read_workspace'}))
    return ('RUN_TESTS',)
