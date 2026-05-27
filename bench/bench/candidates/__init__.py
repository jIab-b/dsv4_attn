"""Built-in candidate implementations for the HCA / CSA benchmark.

Each candidate is a Python module that exposes:
    NAME: str                               # short id used by the CLI
    VARIANT: str                            # 'hca' or 'csa'
    def candidate(weights, inputs, *, spec) -> torch.Tensor: ...

Built-ins:
    hca_torch_eager   - alias for the fp32 reference (sanity test)
    csa_torch_eager   - alias for the fp32 reference (sanity test)
    hca_torch_compiled - reference under torch.compile (default mode)
    csa_torch_compiled - reference under torch.compile (default mode)

User candidates: drop a new module here matching the same interface
(NAME, VARIANT, candidate) and the CLI will pick it up.
"""
import re
from importlib import import_module
from pathlib import Path

_HERE = Path(__file__).parent
_NAME_RE = re.compile(r"""^\s*NAME\s*=\s*["']([^"']+)["']""", re.MULTILINE)
_VARIANT_RE = re.compile(r"""^\s*VARIANT\s*=\s*["']([^"']+)["']""", re.MULTILINE)


def list_candidates() -> dict[str, dict]:
    """Discover candidate modules in this package by *parsing* the source
    (no imports → works even on a host that lacks torch).

    Returns:
        Mapping of NAME → {variant, module_name}.
    """
    out: dict[str, dict] = {}
    for path in sorted(_HERE.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        text = path.read_text()
        m_name = _NAME_RE.search(text)
        m_variant = _VARIANT_RE.search(text)
        if not (m_name and m_variant):
            continue
        out[m_name.group(1)] = {
            "variant": m_variant.group(1),
            "module_name": f"bench.candidates.{path.stem}",
        }
    return out


def load_candidate(name: str):
    """Look up a candidate module by its declared NAME."""
    table = list_candidates()
    if name not in table:
        raise KeyError(f"unknown candidate {name!r}; have: {list(table)}")
    return import_module(table[name]["module_name"])
