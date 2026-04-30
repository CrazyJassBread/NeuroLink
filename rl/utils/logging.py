from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    total_reward: float
    length: int
    terminated: bool
    truncated: bool
    game_over: bool


def write_episode_results_jsonl(path: str | Path, results: Iterable[EpisodeResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
