from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.core.constants import GRID_HEIGHT, GRID_WIDTH
from nesylink.core.world.rooms import RoomManager


ASCII_TILE_TO_LAYOUT = {
    "#": "#",
    ".": ".",
    "P": ".",
    "K": ".",
    "D": ".",
    "M": ".",
    "G": ".",
    "T": ".",
    "E": ".",
}
SUPPORTED_SOURCE_TILES = frozenset(ASCII_TILE_TO_LAYOUT)
DEFAULT_EXIT_MESSAGES = {
    "blocked_locked": "NEED KEY",
    "blocked_monsters": "DEFEAT ALL MONSTERS",
    "success": "CLEARED!",
}


class ExportValidationError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert editable NesyLink map sources (ASCII txt / JSON grid / optional YAML) "
            "into the current pure-room JSON schema."
        )
    )
    parser.add_argument("--input", required=True, help="Source map file path.")
    parser.add_argument("--output", required=True, help="Target JSON file path.")
    parser.add_argument("--room-id", required=True, help="Room id for the generated room JSON.")
    return parser.parse_args()


def load_source(path: Path) -> tuple[dict[str, Any], list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _load_yaml_source(path)
    if suffix == ".json":
        return _load_json_source(path)
    return _load_ascii_source(path)


def _load_yaml_source(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ExportValidationError(
            "YAML source support requires PyYAML at development time. "
            "Runtime map loading does not depend on YAML."
        ) from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExportValidationError(f"{path} must contain an object with 'grid' and optional 'metadata'")
    return _coerce_metadata_and_grid(path, payload)


def _load_json_source(path: Path) -> tuple[dict[str, Any], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return {}, normalize_grid(payload, source_path=path)
    if isinstance(payload, dict):
        return _coerce_metadata_and_grid(path, payload)
    raise ExportValidationError(f"{path} must contain either a grid array or an object with 'grid'")


def _load_ascii_source(path: Path) -> tuple[dict[str, Any], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ExportValidationError(f"{path} is empty")
    return {}, normalize_grid(lines, source_path=path)


def _coerce_metadata_and_grid(path: Path, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    raw_grid = payload.get("grid")
    if raw_grid is None:
        raise ExportValidationError(f"{path} is missing required 'grid'")
    raw_metadata = payload.get("metadata")
    if raw_metadata is None:
        metadata = {key: value for key, value in payload.items() if key != "grid"}
    elif isinstance(raw_metadata, dict):
        metadata = dict(raw_metadata)
    else:
        raise ExportValidationError(f"{path} field 'metadata' must be an object")
    return metadata, normalize_grid(raw_grid, source_path=path)


def normalize_grid(raw_grid: Any, *, source_path: Path) -> list[str]:
    if not isinstance(raw_grid, list) or not raw_grid:
        raise ExportValidationError(f"{source_path} grid must be a non-empty list")

    rows: list[str] = []
    for row_index, row in enumerate(raw_grid):
        if isinstance(row, str):
            rows.append(row)
            continue
        if isinstance(row, list):
            cell_values: list[str] = []
            for col_index, cell in enumerate(row):
                if not isinstance(cell, str) or len(cell) != 1:
                    raise ExportValidationError(
                        f"{source_path} grid[{row_index}][{col_index}] must be a single-character string"
                    )
                cell_values.append(cell)
            rows.append("".join(cell_values))
            continue
        raise ExportValidationError(f"{source_path} grid[{row_index}] must be a string or a list of characters")
    return rows


def validate_grid(grid: list[str], *, source_path: Path) -> list[str]:
    row_widths = {len(row) for row in grid}
    if len(row_widths) != 1:
        raise ExportValidationError(f"{source_path} map must be rectangular")
    width = row_widths.pop()
    if width != GRID_WIDTH or len(grid) != GRID_HEIGHT:
        raise ExportValidationError(
            f"{source_path} must be exactly {GRID_WIDTH}x{GRID_HEIGHT} to match the current loader"
        )

    player_count = 0
    for row_index, row in enumerate(grid):
        for col_index, tile in enumerate(row):
            if tile not in SUPPORTED_SOURCE_TILES:
                allowed = "".join(sorted(SUPPORTED_SOURCE_TILES))
                raise ExportValidationError(
                    f"{source_path} has unknown tile '{tile}' at ({col_index}, {row_index}); allowed: {allowed}"
                )
            if tile == "P":
                player_count += 1
    if player_count != 1:
        raise ExportValidationError(f"{source_path} must contain exactly one player spawn 'P'")
    return grid


def export_map(source_path: Path, output_path: Path, room_id: str) -> dict[str, Any]:
    metadata, raw_grid = load_source(source_path)
    grid = validate_grid(raw_grid, source_path=source_path)
    payload = build_map_payload(grid=grid, metadata=metadata, room_id=room_id)
    validate_export_payload(payload, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return payload


def build_map_payload(*, grid: list[str], metadata: dict[str, Any], room_id: str) -> dict[str, Any]:
    spawn: tuple[int, int] | None = None
    objects: list[dict[str, Any]] = []
    exits: list[dict[str, Any]] = []

    for row_index, row in enumerate(grid):
        for col_index, tile in enumerate(row):
            pos = [col_index, row_index]
            if tile == "P":
                spawn = (col_index, row_index)
            elif tile == "K":
                objects.append(
                    {
                        "id": f"chest_key_{_count_kind(objects, 'chest') + 1}",
                        "kind": "chest",
                        "pos": pos,
                        "loot": {"kind": "key", "key_id": f"{room_id}_key"},
                    }
                )
            elif tile == "G":
                objects.append(
                    {
                        "id": f"chest_gold_{_count_kind(objects, 'chest') + 1}",
                        "kind": "chest",
                        "pos": pos,
                        "loot": {"kind": "gold", "amount": 1},
                    }
                )
            elif tile == "M":
                objects.append(
                    {
                        "id": f"monster_{_count_kind(objects, 'monster') + 1}",
                        "kind": "monster",
                        "pos": pos,
                        "patrol": [[col_index, row_index]],
                    }
                )
            elif tile == "T":
                objects.append(
                    {
                        "id": f"trap_{_count_kind(objects, 'trap') + 1}",
                        "kind": "trap",
                        "pos": pos,
                        "damage": 1,
                        "respawn_to": "default",
                    }
                )
            elif tile in {"D", "E"}:
                exits.append(build_exit(tile=tile, pos=(col_index, row_index), room_id=room_id, metadata=metadata))

    if spawn is None:
        raise ExportValidationError("player spawn 'P' is required")

    if not exits:
        raise ExportValidationError("at least one exit marker 'D' or 'E' is required")

    coord = metadata.get("coord", [0, 0])
    if not isinstance(coord, list) or len(coord) != 2:
        coord = [0, 0]

    return {
        "id": room_id,
        "coord": [int(coord[0]), int(coord[1])],
        "layout": ["".join(ASCII_TILE_TO_LAYOUT[cell] for cell in row) for row in grid],
        "spawns": {"default": [spawn[0], spawn[1]]},
        "default_spawn": "default",
        "objects": objects,
        "exits": exits,
    }


def build_exit(*, tile: str, pos: tuple[int, int], room_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    col_index, row_index = pos
    direction = _edge_direction(col_index, row_index)
    target_room = str(metadata.get("target_room", room_id))
    target_entry = str(metadata.get("target_entry", _entry_name_for_direction(direction)))
    exit_id = str(metadata.get(f"{direction}_exit_id", f"{direction}_exit"))

    if tile == "D":
        return {
            "id": exit_id,
            "direction": direction,
            "target_room": target_room,
            "target_entry": target_entry,
            "type": "locked_key",
            "blocked_message": DEFAULT_EXIT_MESSAGES["blocked_locked"],
            "success_message": DEFAULT_EXIT_MESSAGES["success"],
            "requires": {"key_count": 1, "consume_key": True},
        }

    return {
        "id": exit_id,
        "direction": direction,
        "target_room": target_room,
        "target_entry": target_entry,
        "type": "normal",
        "success_message": DEFAULT_EXIT_MESSAGES["success"],
    }


def validate_export_payload(payload: dict[str, Any], output_path: Path) -> None:
    for forbidden_key in (
        "task",
        "task_id",
        "task_type",
        "objective",
        "progress",
        "reward",
        "rewards",
        "reward_profile",
        "success_condition",
        "failure_condition",
    ):
        if forbidden_key in payload:
            raise ExportValidationError(f"exported payload unexpectedly contains forbidden field '{forbidden_key}'")

    with tempfile.TemporaryDirectory(prefix="nesylink_export_") as tmp_dir:
        temp_output = Path(tmp_dir) / output_path.name
        temp_output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        RoomManager(temp_output)


def _count_kind(objects: list[dict[str, Any]], kind: str) -> int:
    return sum(1 for obj in objects if obj.get("kind") == kind)


def _edge_direction(col_index: int, row_index: int) -> str:
    if row_index == 0:
        return "north"
    if row_index == GRID_HEIGHT - 1:
        return "south"
    if col_index == 0:
        return "west"
    if col_index == GRID_WIDTH - 1:
        return "east"
    raise ExportValidationError("exit markers 'D' and 'E' must be placed on the room edge")


def _entry_name_for_direction(direction: str) -> str:
    return {
        "north": "from_south",
        "south": "from_north",
        "west": "from_east",
        "east": "from_west",
    }[direction]


def main() -> int:
    args = parse_args()
    try:
        payload = export_map(
            source_path=Path(args.input),
            output_path=Path(args.output),
            room_id=str(args.room_id),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"output": str(args.output), "room_id": payload["id"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
