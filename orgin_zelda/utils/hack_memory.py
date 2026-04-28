from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pyboy import PyBoy


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from envs.config import (
	ADDR_CUR_HEALTH,
	ADDR_KEYS,
	ADDR_MAX_HEALTH,
	ADDR_ROOM_ID,
	ADDR_RUPEE,
)


KNOWN_ADDRS = {
	"cur_health": ADDR_CUR_HEALTH,
	"max_health": ADDR_MAX_HEALTH,
	"rupee": ADDR_RUPEE,
	"room_id": ADDR_ROOM_ID,
	"keys": ADDR_KEYS,
}


def _parse_address(token: str) -> int:
	name = token.strip().lower()
	if name in KNOWN_ADDRS:
		return KNOWN_ADDRS[name]
	return int(token, 0)


def _parse_byte(token: str) -> int:
	value = int(token, 0)
	if value < 0 or value > 255:
		raise ValueError(f"byte value must be in [0, 255], got {value}")
	return value


def _parse_freeze_spec(spec: str) -> tuple[int, int]:
	if "=" not in spec:
		raise ValueError(f"freeze spec must be address=value, got: {spec}")
	left, right = spec.split("=", 1)
	return _parse_address(left), _parse_byte(right)


def _fmt_addr(addr: int) -> str:
	return f"0x{addr:04X}"


def _read_range(pyboy: "PyBoy", start: int, end: int) -> list[int]:
	return [int(pyboy.memory[addr]) for addr in range(start, end + 1)]


def _save_snapshot(path: Path, start: int, end: int, values: list[int]) -> None:
	payload = {
		"start": start,
		"end": end,
		"values": values,
	}
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload), encoding="utf-8")


def _load_snapshot(path: Path) -> tuple[int, int, list[int]]:
	data = json.loads(path.read_text(encoding="utf-8"))
	start = int(data["start"])
	end = int(data["end"])
	values = [int(v) for v in data["values"]]
	if len(values) != (end - start + 1):
		raise ValueError("invalid snapshot: values length does not match address range")
	return start, end, values


def _build_filter(mode: str):
	if mode == "changed":
		return lambda before, after: after != before
	if mode == "increased":
		return lambda before, after: after > before
	if mode == "decreased":
		return lambda before, after: after < before
	if mode == "unchanged":
		return lambda before, after: after == before
	raise ValueError(f"unsupported filter mode: {mode}")


def _print_known_addresses() -> None:
	print("Known symbols:")
	for name, addr in KNOWN_ADDRS.items():
		print(f"  {name:<12} -> {_fmt_addr(addr)}")


def _print_quick_examples() -> None:
	print("\nQuick examples:")
	print("  python utils/hack_memory.py --state game_state/Room_51.state peek cur_health rupee")
	print("  python utils/hack_memory.py --state game_state/Room_51.state poke rupee 99")
	print("  python utils/hack_memory.py --state game_state/Room_51.state scan --init")
	print("  python utils/hack_memory.py --state game_state/Room_51.state scan --filter decreased --update-snapshot")


def _load_state_if_exists(pyboy: "PyBoy", state_path: str | None) -> None:
	if not state_path:
		return
	path = Path(state_path)
	if not path.exists():
		print(f"State not found, skip load: {path}")
		return
	with path.open("rb") as handle:
		pyboy.load_state(handle)
	print(f"Loaded state: {path}")



def _open_pyboy(game: str, window: str) -> "PyBoy":
	from pyboy import PyBoy

	return PyBoy(game, sound_emulated=False, window=window)


def run_peek(args: argparse.Namespace) -> None:
	pyboy = _open_pyboy(args.game, args.window)
	try:
		_load_state_if_exists(pyboy, args.state)
		for token in args.addresses:
			addr = _parse_address(token)
			value = int(pyboy.memory[addr])
			print(f"{token:>12} ({_fmt_addr(addr)}): {value}")
	finally:
		pyboy.stop()


def run_poke(args: argparse.Namespace) -> None:
	pyboy = _open_pyboy(args.game, args.window)
	try:
		_load_state_if_exists(pyboy, args.state)
		addr = _parse_address(args.address)
		value = _parse_byte(args.value)
		before = int(pyboy.memory[addr])
		pyboy.memory[addr] = value
		after = int(pyboy.memory[addr])
		print(f"Write {_fmt_addr(addr)}: {before} -> {after}")

		if args.save_state:
			save_path = Path(args.save_state)
			with save_path.open("wb") as handle:
				pyboy.save_state(handle)
			print(f"Saved state: {save_path}")
	finally:
		pyboy.stop()


def run_watch(args: argparse.Namespace) -> None:
	pyboy = _open_pyboy(args.game, args.window)
	try:
		_load_state_if_exists(pyboy, args.state)
		addrs = [_parse_address(token) for token in args.addresses]
		print("Press Ctrl+C to stop watch.")
		frame = 0
		while True:
			pyboy.tick(1)
			frame += 1
			if frame % args.every == 0:
				values = [int(pyboy.memory[a]) for a in addrs]
				rendered = " ".join(
					f"{_fmt_addr(a)}={v:3d}" for a, v in zip(addrs, values)
				)
				print(f"frame={frame:6d} {rendered}")
			if args.sleep > 0:
				time.sleep(args.sleep)
	except KeyboardInterrupt:
		print("Watch stopped.")
	finally:
		pyboy.stop()


def run_freeze(args: argparse.Namespace) -> None:
	pyboy = _open_pyboy(args.game, args.window)
	try:
		_load_state_if_exists(pyboy, args.state)
		frozen = dict(_parse_freeze_spec(spec) for spec in args.set)
		print("Freeze list:")
		for addr, value in frozen.items():
			print(f"  {_fmt_addr(addr)} -> {value}")

		print("Press Ctrl+C to stop freeze.")
		frame = 0
		while True:
			for addr, value in frozen.items():
				pyboy.memory[addr] = value

			pyboy.tick(1)
			frame += 1
			if frame % args.every == 0:
				rendered = " ".join(
					f"{_fmt_addr(a)}={int(pyboy.memory[a]):3d}" for a in frozen
				)
				print(f"frame={frame:6d} {rendered}")

			if args.sleep > 0:
				time.sleep(args.sleep)
	except KeyboardInterrupt:
		print("Freeze stopped.")
	finally:
		pyboy.stop()


def run_scan(args: argparse.Namespace) -> None:
	pyboy = _open_pyboy(args.game, args.window)
	try:
		_load_state_if_exists(pyboy, args.state)

		snapshot_path = Path(args.snapshot) if args.snapshot else None
		start = _parse_address(args.start)
		end = _parse_address(args.end)
		if end < start:
			raise ValueError("end address must be >= start address")

		if args.init:
			base_values = _read_range(pyboy, start, end)
			if snapshot_path is None:
				raise ValueError("--init requires --snapshot")
			_save_snapshot(snapshot_path, start, end, base_values)
			print(f"Baseline snapshot saved: {snapshot_path}")
			print(f"Range: {_fmt_addr(start)}-{_fmt_addr(end)} ({len(base_values)} bytes)")
			return

		if snapshot_path is not None and snapshot_path.exists():
			snap_start, snap_end, base_values = _load_snapshot(snapshot_path)
			start, end = snap_start, snap_end
		else:
			base_values = _read_range(pyboy, start, end)
			if snapshot_path is not None:
				_save_snapshot(snapshot_path, start, end, base_values)
				print(f"Baseline snapshot created: {snapshot_path}")
				print("Do your in-game action, then run the same scan command again.")
				return

		if args.ticks > 0:
			print(f"Ticking {args.ticks} frames. Do your in-game action now...")
			for _ in range(args.ticks):
				pyboy.tick(1)
				if args.sleep > 0:
					time.sleep(args.sleep)

		current_values = _read_range(pyboy, start, end)
		predicate = _build_filter(args.filter)

		matches: list[tuple[int, int, int, int]] = []
		for idx, (before, after) in enumerate(zip(base_values, current_values)):
			if not predicate(before, after):
				continue
			if args.value is not None and after != args.value:
				continue
			addr = start + idx
			matches.append((addr, before, after, after - before))

		print(
			f"Scan range: {_fmt_addr(start)}-{_fmt_addr(end)} | "
			f"filter={args.filter} | matches={len(matches)}"
		)

		for addr, before, after, delta in matches[: args.top]:
			print(f"{_fmt_addr(addr)}: {before:3d} -> {after:3d} (delta {delta:+d})")

		if args.results:
			results_path = Path(args.results)
			results_path.parent.mkdir(parents=True, exist_ok=True)
			with results_path.open("w", encoding="utf-8") as handle:
				handle.write("address,before,after,delta\n")
				for addr, before, after, delta in matches:
					handle.write(f"{_fmt_addr(addr)},{before},{after},{delta}\n")
			print(f"Saved matches: {results_path}")

		if snapshot_path is not None and args.update_snapshot:
			_save_snapshot(snapshot_path, start, end, current_values)
			print(f"Updated snapshot: {snapshot_path}")

	except KeyboardInterrupt:
		print("Scan interrupted.")
	finally:
		pyboy.stop()


def _print_hunt_help() -> None:
	print("Hunt commands (press Enter after command):")
	print("  s           keep addresses that changed")
	print("  i           keep addresses that increased")
	print("  d           keep addresses that decreased")
	print("  u           keep addresses that stayed unchanged")
	print("  v <0-255>   keep addresses whose current value == v")
	print("  r           reset baseline to current memory")
	print("  p           print top current candidates")
	print("  q           quit hunt")


def _render_candidates(
	start: int,
	baseline: list[int],
	current: list[int],
	candidates: list[int],
	top: int,
) -> None:
	rows: list[tuple[int, int, int, int]] = []
	for offset in candidates:
		before = baseline[offset]
		after = current[offset]
		rows.append((start + offset, before, after, after - before))

	rows.sort(key=lambda row: abs(row[3]), reverse=True)
	for addr, before, after, delta in rows[:top]:
		print(f"{_fmt_addr(addr)}: {before:3d} -> {after:3d} (delta {delta:+d})")


def run_hunt(args: argparse.Namespace) -> None:
	pyboy = _open_pyboy(args.game, args.window)
	try:
		_load_state_if_exists(pyboy, args.state)
		start = _parse_address(args.start)
		end = _parse_address(args.end)
		if end < start:
			raise ValueError("end address must be >= start address")

		width = end - start + 1
		baseline = _read_range(pyboy, start, end)
		candidates = list(range(width))

		print(f"Live hunt range: {_fmt_addr(start)}-{_fmt_addr(end)} ({width} bytes)")
		print("Play in SDL2 window, then use terminal commands to narrow addresses.")
		_print_hunt_help()

		frame = 0
		last_status = 0
		running = True
		while running:
			pyboy.tick(1)
			frame += 1

			if args.sleep > 0:
				time.sleep(args.sleep)

			if args.status_every > 0 and (frame - last_status) >= args.status_every:
				print(f"frame={frame:7d} candidates={len(candidates)}")
				last_status = frame

			ready, _, _ = select.select([sys.stdin], [], [], 0)
			if not ready:
				continue

			command_line = sys.stdin.readline().strip()
			if not command_line:
				continue

			parts = command_line.split()
			cmd = parts[0].lower()
			current = _read_range(pyboy, start, end)

			if cmd == "q":
				print("Hunt stopped.")
				running = False
				continue

			if cmd == "r":
				baseline = current
				print("Baseline reset to current memory.")
				continue

			if cmd == "p":
				print(f"Candidates: {len(candidates)}")
				_render_candidates(start, baseline, current, candidates, args.top)
				continue

			if cmd == "v":
				if len(parts) != 2:
					print("Usage: v <0-255>")
					continue
				wanted = _parse_byte(parts[1])
				candidates = [offset for offset in candidates if current[offset] == wanted]
				print(f"Applied value filter == {wanted}. candidates={len(candidates)}")
				_render_candidates(start, baseline, current, candidates, args.top)
				if args.step_baseline:
					baseline = current
				continue

			mode_map = {
				"s": "changed",
				"i": "increased",
				"d": "decreased",
				"u": "unchanged",
			}
			if cmd not in mode_map:
				print(f"Unknown command: {cmd}")
				_print_hunt_help()
				continue

			predicate = _build_filter(mode_map[cmd])
			candidates = [
				offset
				for offset in candidates
				if predicate(baseline[offset], current[offset])
			]
			print(f"Applied {mode_map[cmd]}. candidates={len(candidates)}")
			_render_candidates(start, baseline, current, candidates, args.top)

			if args.step_baseline:
				baseline = current

	except KeyboardInterrupt:
		print("Hunt interrupted.")
	finally:
		pyboy.stop()


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="PyBoy memory utility: peek, poke, watch, freeze, scan, hunt"
	)
	parser.add_argument(
		"--game",
		default="game_state/Link's awakening.gb",
		help="Path to ROM file",
	)
	parser.add_argument(
		"--state",
		default=None,
		help="Optional state file to load before operation",
	)
	parser.add_argument(
		"--window",
		choices=["SDL2", "null"],
		default="SDL2",
		help="PyBoy window mode; use null for headless",
	)

	sub = parser.add_subparsers(dest="command", required=False)

	p_peek = sub.add_parser("peek", help="Read one or more memory addresses")
	p_peek.add_argument("addresses", nargs="+", help="address or symbol name")
	p_peek.set_defaults(func=run_peek)

	p_poke = sub.add_parser("poke", help="Write one byte to memory")
	p_poke.add_argument("address", help="address or symbol name")
	p_poke.add_argument("value", help="byte value (0-255)")
	p_poke.add_argument(
		"--save-state",
		default=None,
		help="Optional path to save state after write",
	)
	p_poke.set_defaults(func=run_poke)

	p_watch = sub.add_parser("watch", help="Watch addresses while game runs")
	p_watch.add_argument("addresses", nargs="+", help="address or symbol name")
	p_watch.add_argument("--every", type=int, default=30, help="print every N frames")
	p_watch.add_argument("--sleep", type=float, default=0.0, help="sleep seconds each frame")
	p_watch.set_defaults(func=run_watch)

	p_freeze = sub.add_parser("freeze", help="Force values every frame")
	p_freeze.add_argument(
		"--set",
		action="append",
		required=True,
		help="address=value or symbol=value; can be repeated",
	)
	p_freeze.add_argument("--every", type=int, default=30, help="print every N frames")
	p_freeze.add_argument("--sleep", type=float, default=0.0, help="sleep seconds each frame")
	p_freeze.set_defaults(func=run_freeze)

	p_scan = sub.add_parser("scan", help="Unknown address scanner with snapshots")
	p_scan.add_argument(
		"--snapshot",
		default="game_state/memscan_snapshot.json",
		help="Snapshot file path used across scan rounds",
	)
	p_scan.add_argument(
		"--init",
		action="store_true",
		help="Capture baseline snapshot and exit",
	)
	p_scan.add_argument(
		"--start",
		default="0xC000",
		help="start address for scan range (ignored when snapshot exists)",
	)
	p_scan.add_argument(
		"--end",
		default="0xDFFF",
		help="end address for scan range (ignored when snapshot exists)",
	)
	p_scan.add_argument(
		"--filter",
		choices=["changed", "increased", "decreased", "unchanged"],
		default="changed",
		help="comparison mode against baseline snapshot",
	)
	p_scan.add_argument(
		"--value",
		type=int,
		default=None,
		help="optional exact current value filter (0-255)",
	)
	p_scan.add_argument(
		"--ticks",
		type=int,
		default=0,
		help="frames to run before sampling current memory",
	)
	p_scan.add_argument(
		"--sleep",
		type=float,
		default=0.0,
		help="optional sleep per tick during --ticks",
	)
	p_scan.add_argument(
		"--top",
		type=int,
		default=200,
		help="show only the first N matches",
	)
	p_scan.add_argument(
		"--results",
		default=None,
		help="optional CSV output path for all matches",
	)
	p_scan.add_argument(
		"--update-snapshot",
		action="store_true",
		help="replace snapshot with current sample for next narrowing round",
	)
	p_scan.set_defaults(func=run_scan)

	p_hunt = sub.add_parser(
		"hunt",
		help="Interactive live hunt for unknown addresses while manually playing",
	)
	p_hunt.add_argument(
		"--start",
		default="0xC000",
		help="start address for live hunt",
	)
	p_hunt.add_argument(
		"--end",
		default="0xDFFF",
		help="end address for live hunt",
	)
	p_hunt.add_argument(
		"--top",
		type=int,
		default=20,
		help="print top N candidates each step",
	)
	p_hunt.add_argument(
		"--status-every",
		type=int,
		default=300,
		help="print periodic status every N frames (0 to disable)",
	)
	p_hunt.add_argument(
		"--sleep",
		type=float,
		default=0.0,
		help="optional sleep per frame to reduce CPU",
	)
	p_hunt.add_argument(
		"--step-baseline",
		action="store_true",
		help="after each filter command, update baseline to current values",
	)
	p_hunt.set_defaults(func=run_hunt)

	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	if not hasattr(args, "func"):
		parser.print_help()
		_print_quick_examples()
		return
	_print_known_addresses()
	args.func(args)


if __name__ == "__main__":
	main()
