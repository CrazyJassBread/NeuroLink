from __future__ import annotations

import math
import sys
from pathlib import Path

from env_diy.env import make_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DUNGEON_ROOT = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons"
CONFIGS = [
    DUNGEON_ROOT / "prototype" / "dungeon.json",
    DUNGEON_ROOT / "combat_training" / "dungeon.json",
    DUNGEON_ROOT / "evasion_training" / "dungeon.json",
    DUNGEON_ROOT / "chest_training" / "dungeon.json",
    DUNGEON_ROOT / "avoid_traps" / "room_001.json",
    DUNGEON_ROOT / "kill_monsters" / "room_001.json",
    DUNGEON_ROOT / "key_door" / "room_001.json",
]


def main() -> int:
    failures: list[str] = []
    for index, config_path in enumerate(CONFIGS):
        env = make_env(config_path, api="gym")
        total_reward = 0.0
        success = False
        failure = False
        terminated_reason = None
        try:
            obs, info = env.reset(seed=index)
            env.action_space.seed(index)
            for step in range(100):
                action = int(env.action_space.sample())
                obs, reward, terminated, truncated, info = env.step(action)
                if math.isnan(float(reward)):
                    raise RuntimeError("reward is NaN")
                total_reward += float(reward)
                task_info = info["task"]
                success = bool(task_info.get("success", False))
                failure = bool(task_info.get("failure", False))
                terminated_reason = task_info.get("terminated_reason")
                if terminated or truncated:
                    break
            print(
                f"{config_path.name}: return={total_reward:.3f} success={success} "
                f"failure={failure} terminated_reason={terminated_reason}"
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{config_path}: {exc}")
        finally:
            env.close()
    if failures:
        for failure_line in failures:
            print(f"ERROR {failure_line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
