from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


FORBIDDEN_MAP_KEYS = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate a legacy nesylink room JSON into the pure map schema.")
    parser.add_argument("--input", required=True, help="Legacy room JSON path.")
    parser.add_argument("--output", required=True, help="Migrated pure room JSON path.")
    parser.add_argument(
        "--report",
        required=False,
        default=None,
        help="Optional side report JSON path containing removed legacy task metadata.",
    )
    return parser.parse_args()


def migrate_map_schema(input_path: Path, output_path: Path, report_path: Path | None = None) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{input_path} must contain a JSON object")

    migrated = dict(payload)
    room_id = str(migrated.pop("id", migrated.pop("room_id", "room_001")))
    coord = migrated.get("coord")
    if not isinstance(coord, list) or len(coord) != 2:
        coord = [0, 0]
    migrated["id"] = room_id
    migrated["coord"] = [int(coord[0]), int(coord[1])]

    legacy_task = {key: payload[key] for key in FORBIDDEN_MAP_KEYS if key in payload}
    for key in FORBIDDEN_MAP_KEYS:
        migrated.pop(key, None)
    migrated.pop("room_id", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(migrated, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if report_path is not None:
        report = {
            "source": str(input_path),
            "output": str(output_path),
            "legacy_task": legacy_task,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    return migrated


def main() -> int:
    args = parse_args()
    try:
        migrate_map_schema(
            input_path=Path(args.input),
            output_path=Path(args.output),
            report_path=None if args.report is None else Path(args.report),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
