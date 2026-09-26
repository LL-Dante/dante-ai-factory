"""Compatibility entry point for the authenticated production Node 0 queue CLI."""
from __future__ import annotations

import sys

from dante.node0_cli import main as node0_main


def main(argv=None):
    return node0_main(["workload", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
