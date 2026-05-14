from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_MAP_PATHS = (
    PROJECT_ROOT / "env_diy" / "maps",
    PROJECT_ROOT / "env_diy" / "map_data" / "dungeons",
)


def load_map(*, map_id: str | None = None, map_path: str | Path | None = None) -> Path:
    if map_path is not None:
        return _resolve_path(map_path)
    if map_id is None or not str(map_id).strip():
        raise ValueError("either map_id or map_path must be provided")

    normalized = str(map_id).strip()
    candidates = [
        BUILTIN_MAP_PATHS[0] / f"{normalized}.json",
        BUILTIN_MAP_PATHS[1] / normalized / "dungeon.json",
        BUILTIN_MAP_PATHS[1] / f"{normalized}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise ValueError(f"unknown map_id '{normalized}', searched: {searched}")


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"map path does not exist: {candidate}")
    return candidate
