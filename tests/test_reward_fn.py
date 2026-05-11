from __future__ import annotations

import unittest

from env_diy.core.types import EngineStepResult, RuntimeSnapshot, StuckPenaltyConfig
from env_diy.reward.compute import compute_reward as shim_compute_reward
from env_diy.rewards.reward_fn import RewardConfig, compute_reward


def _snapshot(
    *,
    room_id: str = "room",
    pos: tuple[float, float] = (16.0, 16.0),
    health: int = 3,
    gold: int = 0,
    keys: int = 0,
    task_finished: bool = False,
    no_progress_steps: int = 0,
    step_count: int = 0,
    episode_id: int = 0,
    seed: int | None = 0,
) -> RuntimeSnapshot:
    return RuntimeSnapshot(
        room_id=room_id,
        room_coord=(0, 0),
        player_position_px=pos,
        player_tile=(1, 1),
        health=health,
        gold=gold,
        keys=keys,
        items=(),
        no_progress_steps=no_progress_steps,
        task_finished=task_finished,
        step_count=step_count,
        episode_id=episode_id,
        seed=seed,
    )


class RewardFunctionTests(unittest.TestCase):
    def test_legacy_reward_parity_for_key_events(self) -> None:
        cases = [
            ("movement", EngineStepResult(events=["move_right"]), -0.01),
            ("empty_action", EngineStepResult(events=["action_a_empty"]), -0.01),
            ("blocked_exit", EngineStepResult(events=["blocked_locked"]), -0.02),
            ("picked_key", EngineStepResult(events=["got_key"]), 0.4),
            (
                "opened_door",
                EngineStepResult(
                    events=["door_unlocked"],
                    event_details=[{"type": "door_unlocked", "direction": "north", "key_consumed": True}],
                ),
                0.0,
            ),
            ("picked_coin", EngineStepResult(events=["got_gold"]), 0.2),
            ("killed_monster", EngineStepResult(events=["monster_killed"]), 0.3),
            ("hit_trap", EngineStepResult(events=["trap_damage"]), -0.5),
            ("agent_dead", EngineStepResult(events=["game_over"], terminated=True), 0.0),
            ("task_finished", EngineStepResult(events=["task_finished"]), 10.0),
        ]
        prev = _snapshot()
        next_state = _snapshot()
        stuck = StuckPenaltyConfig(enabled=False, steps=30, reward=-0.01)

        for label, result, expected in cases:
            with self.subTest(label=label):
                shim_reward, shim_terms = shim_compute_reward(prev, next_state, result, None, stuck)
                canonical_reward, canonical_terms = compute_reward(
                    prev,
                    next_state,
                    result,
                    task_spec=None,
                    config=RewardConfig(reward_mode="legacy", stuck_penalty=stuck),
                )
                self.assertAlmostEqual(shim_reward, expected)
                self.assertAlmostEqual(canonical_reward, expected)
                self.assertEqual(shim_terms, canonical_terms)
                self.assertAlmostEqual(canonical_reward, sum(canonical_terms.values()))

    def test_sparse_reward_only_rewards_goal(self) -> None:
        prev = _snapshot()
        next_state = _snapshot(task_finished=True)
        reward, terms = compute_reward(
            prev,
            next_state,
            EngineStepResult(events=["task_finished"]),
            config=RewardConfig(reward_mode="sparse"),
        )
        self.assertEqual(reward, 10.0)
        self.assertEqual(terms, {"reached_goal": 10.0})

        reward, terms = compute_reward(
            prev,
            prev,
            EngineStepResult(events=["move_right"]),
            config=RewardConfig(reward_mode="sparse"),
        )
        self.assertEqual(reward, 0.0)
        self.assertEqual(terms, {})

    def test_event_reward_uses_named_terms(self) -> None:
        prev = _snapshot()
        next_state = _snapshot(health=2)
        reward, terms = compute_reward(
            prev,
            next_state,
            EngineStepResult(events=["move_right", "got_key", "door_unlocked", "trap_damage"]),
            config=RewardConfig(reward_mode="event"),
        )
        self.assertIn("step_penalty", terms)
        self.assertIn("picked_key", terms)
        self.assertIn("opened_door", terms)
        self.assertIn("hit_trap", terms)
        self.assertAlmostEqual(reward, sum(terms.values()))


if __name__ == "__main__":
    unittest.main()
