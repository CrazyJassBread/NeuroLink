from __future__ import annotations

import unittest

from env_diy.core.events import (
    event_counts_to_flags,
    event_records_to_counts,
    merge_event_counts,
    merge_event_records,
    normalize_event_records,
)


class EventHelperTests(unittest.TestCase):
    def test_event_records_to_counts_and_flags(self) -> None:
        records = normalize_event_records(
            ["move_right", "move_right", "got_key"],
            [{"type": "door_unlocked", "direction": "east"}],
        )
        counts = event_records_to_counts(records)
        flags = event_counts_to_flags(counts)
        self.assertEqual(counts["move_right"], 2)
        self.assertEqual(counts["got_key"], 1)
        self.assertEqual(counts["door_unlocked"], 1)
        self.assertTrue(flags["got_key"])

    def test_merge_event_counts_accumulates(self) -> None:
        merged = merge_event_counts({"move_right": 1}, {"move_right": 2, "got_key": 1})
        self.assertEqual(merged, {"move_right": 3, "got_key": 1})

    def test_merge_event_records_preserves_order(self) -> None:
        first = [{"name": "move_right"}, {"name": "got_key"}]
        second = [{"name": "door_unlocked"}]
        self.assertEqual(merge_event_records(first, second), first + second)


if __name__ == "__main__":
    unittest.main()
