"""Direct executable entry for Windows Task Scheduler."""
from __future__ import annotations
import sys
import os
import json
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from dante.node0_supervisor import main
except Exception as exc:
    try:
        root = Path(os.environ['LOCALAPPDATA']) / 'DanteNode0'
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'supervisor-startup-errors.jsonl'
        if path.exists() and path.stat().st_size > 256 * 1024:
            old = root / 'supervisor-startup-errors.jsonl.1'
            old.unlink(missing_ok=True)
            os.replace(path, old)
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'event': 'startup.import_failed', 'error_type': type(exc).__name__}) + '\n')
    except Exception:
        pass
    raise
raise SystemExit(main())
