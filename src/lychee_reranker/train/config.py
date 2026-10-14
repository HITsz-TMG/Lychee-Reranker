from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON or YAML configuration without silently accepting an empty file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    if config_path.suffix.lower() == ".json":
        value = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError as error:
            raise ImportError("YAML configs require the 'train' extra (PyYAML)") from error
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("the top-level configuration must be a mapping")
    return value


def require_sections(config: Mapping[str, Any], *sections: str) -> None:
    missing = [name for name in sections if not isinstance(config.get(name), dict)]
    if missing:
        raise ValueError(f"missing configuration section(s): {', '.join(missing)}")


def apply_path_overrides(
    config: Dict[str, Any], *, data: list[str] | None, output_dir: str | None
) -> Dict[str, Any]:
    """Apply the two portable CLI overrides without changing the loaded object."""
    result = {
        key: (dict(value) if isinstance(value, dict) else value) for key, value in config.items()
    }
    if data is not None:
        result.setdefault("data", {})["train_data"] = data
    if output_dir is not None:
        result.setdefault("training", {})["output_dir"] = output_dir
    return result
