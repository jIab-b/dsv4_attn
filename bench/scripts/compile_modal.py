#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

from _common import add_paths


def main() -> int:
    add_paths()
    import modal
    from bench.modal import app, gpu_compile

    with modal.enable_output():
        with app.run():
            result = gpu_compile.remote(sys.argv[1:] or None)
    print(json.dumps(result, indent=2))
    return int(result.get("returncode", 1))


if __name__ == "__main__":
    raise SystemExit(main())
