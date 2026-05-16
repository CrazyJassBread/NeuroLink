# RL Helpers

The RL helpers now build environments by selecting a map plus a reward.

Examples:

```bash
python rl/train_random.py --config nesylink/map_data/dungeons/prototype/dungeon.json
python rl/train_single_task.py --task key_door --episodes 1 --max-steps 50
python rl/train.py --method ppo --task-rooms key_door --total-timesteps 50000
```

Internally the training helpers use:

- `map_path`
- `reward_id`
- `reward_module`
- `max_steps`

They no longer rely on task registry lookups.
