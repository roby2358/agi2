"""Helpers shared by the hegel-lab scripts. Run them from hegel/ (see justfile)."""

import dataclasses
import resource
from typing import Any, List

from src.config import HegelConfig, load_config


def config_named(name: str) -> HegelConfig:
    """Load resources/<name>.toml; a path to a TOML file works too."""
    path = name if name.endswith(".toml") else f"resources/{name}.toml"
    return load_config(path)


def with_setting(config: HegelConfig, key: str, raw: str) -> HegelConfig:
    """A copy of config with one field set from a command-line string."""
    kinds = {f.name: type(getattr(config, f.name)) for f in dataclasses.fields(config)}
    if key not in kinds:
        raise SystemExit(f"No setting '{key}'. Settings: {', '.join(sorted(kinds))}")
    value: Any = kinds[key](raw)
    return dataclasses.replace(config, **{key: value})


def peak_gb() -> float:
    """Peak memory of this process so far, in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def table(header: List[str], rows: List[List[str]]) -> str:
    """Rows as left-aligned columns."""
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in [header] + rows]
    return "\n".join(line.rstrip() for line in lines)
