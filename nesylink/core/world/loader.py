from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BUILTIN_DUNGEON_ROOT = PROJECT_ROOT / "nesylink" / "map_data" / "dungeons"


def load_map(*, map_id: str | None = None, map_path: str | Path | None = None) -> Path:
    if map_path is not None:
        return _resolve_path(map_path)
    if map_id is None or not str(map_id).strip():
        raise ValueError("either map_id or map_path must be provided")

    normalized = str(map_id).strip()
    candidates = []
    if normalized == "dungeon":
        candidates.append(BUILTIN_DUNGEON_ROOT / "prototype" / "dungeon.json")
    candidates.extend(
        [
            BUILTIN_DUNGEON_ROOT / normalized / "dungeon.json",
            BUILTIN_DUNGEON_ROOT / f"{normalized}.json",
        ]
    )
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
        remapped = _remap_legacy_map_path(candidate)
        if remapped is not None:
            candidate = remapped
    if not candidate.exists():
        raise FileNotFoundError(f"map path does not exist: {candidate}")
    return candidate


def _remap_legacy_map_path(candidate: Path) -> Path | None:
    try:
        relative = candidate.relative_to(PROJECT_ROOT)
    except ValueError:
        return None

    if len(relative.parts) != 3 or relative.parts[:2] != ("nesylink", "maps"):
        return None

    map_name = Path(relative.parts[-1]).stem
    if map_name == "dungeon":
        remapped = BUILTIN_DUNGEON_ROOT / "prototype" / "dungeon.json"
        if remapped.exists():
            return remapped.resolve()

    for remapped in (
        BUILTIN_DUNGEON_ROOT / map_name / "dungeon.json",
        BUILTIN_DUNGEON_ROOT / f"{map_name}.json",
    ):
        if remapped.exists():
            return remapped.resolve()
    return None
