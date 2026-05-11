from __future__ import annotations

import unittest
from pathlib import Path

from env_diy.env import make_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRUCTURED_DUNGEON = (
    PROJECT_ROOT / "env_diy" / "map_data" / "dungeons" / "prototype" / "dungeon.json"
)


def _rollout(seed: int, actions: list[int], *, sample_actions: bool) -> tuple[list[float], list[tuple[float, float]], list[dict]]:
    env = make_env(STRUCTURED_DUNGEON, api="gym")
    rewards: list[float] = []
    positions: list[tuple[float, float]] = []
    infos: list[dict] = []
    try:
        obs, info = env.reset(seed=seed)
        if sample_actions:
            env.action_space.seed(seed)
            action_list = [int(env.action_space.sample()) for _ in actions]
        else:
            action_list = list(actions)
        for action in action_list:
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            positions.append(tuple(info["agent_pos"]))
            # Keep only determinism-relevant fields here; episode counters are intentionally diagnostic.
            infos.append(
                {
                    "terminated": terminated,
                    "truncated": truncated,
                    "events": tuple(info["events"]),
                    "seed": info["seed"],
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    return rewards, positions, infos


class SeedDeterminismTests(unittest.TestCase):
    def test_same_seed_and_same_action_sequence_are_reproducible(self) -> None:
        actions = [0, 1, 2, 3, 4, 5, 6, 0, 2, 4]
        first = _rollout(7, actions, sample_actions=False)
        second = _rollout(7, actions, sample_actions=False)
        self.assertEqual(first, second)

    def test_same_seed_and_sampled_actions_are_reproducible(self) -> None:
        action_slots = [0] * 12
        first = _rollout(11, action_slots, sample_actions=True)
        second = _rollout(11, action_slots, sample_actions=True)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
