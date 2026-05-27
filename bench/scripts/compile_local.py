#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys

from _common import REPO_ROOT


def main() -> int:
    cmd = [sys.executable, str(REPO_ROOT / "hca" / "compile.py"), *sys.argv[1:]]
    print(" ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
