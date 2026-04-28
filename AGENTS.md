# AGENTS.md

## Required Reading Order

Before making any code changes, read:

1. `AGENTS.md`
2. `DEVELOPMENT_GUIDE.md`
3. `README.md`
4. Relevant files under `docs/`
5. Relevant source files
6. Relevant tests
7. Relevant map/config examples

Do not modify code before producing a plan and receiving human approval.

---

## Workflow

For every development task:

1. Inspect the current implementation.
2. Identify the minimal required changes.
3. Produce a plan.
4. Wait for approval.
5. Implement changes incrementally.
6. Run smoke tests using the project `.venv`.
7. Fix obvious failures introduced by the changes.
8. Re-run smoke tests.
9. Check code structure and readability.
10. Update documentation.
11. Update worklog.
12. Summarize results.

---

## Planning Requirement

Before editing, provide:

```md
## Plan

### Files to modify
- ...

### Design choices
- ...

### Behavior changes
- ...

### Risks
- ...

### Test plan
- ...

### Rollback plan
- ...
````

Do not edit files until the plan is approved.

---

## Virtual Environment

Use the existing project virtual environment.

Preferred commands:

```bash
source .venv/bin/activate
python --version
python -m pip list
```

If `.venv` does not exist, do not create a new environment unless explicitly asked. Report that `.venv` is missing.

---

## Smoke Test Commands

After code changes, run the most relevant available commands.

Try these in order when applicable:

```bash
source .venv/bin/activate

python -m pytest
python -m pytest tests
python -m pytest -q
```

If the project has lint/type tools configured, also run available commands such as:

```bash
python -m ruff check .
python -m mypy .
python -m pyright
```

Only run tools that are already installed or configured in the project. Do not add new dependencies only for smoke testing unless approved.

---

## Manual Environment Smoke Test

If the environment package can be imported, run a minimal smoke test.

Example:

```bash
source .venv/bin/activate

python - <<'PY'
import gymnasium as gym

# Replace this with the actual environment id if registered.
# env = gym.make("YourEnv-v0")

print("Gymnasium import OK")
PY
```

If the environment is not registered with Gymnasium, run the project’s documented local constructor instead.

The smoke test should check:

* environment can be imported
* environment can be instantiated
* `reset()` returns `(obs, info)`
* `step(action)` returns `(obs, reward, terminated, truncated, info)`
* `render()` does not crash
* `close()` does not crash

---

## Bug Fix Policy After Smoke Test

If smoke tests fail:

1. Determine whether the failure was introduced by the current change.
2. Fix simple, local failures.
3. Re-run the relevant test.
4. Do not perform unrelated large refactors.
5. If failure is unrelated or too large, document it under `Known Issues`.

Do not hide failing tests.

---

## Code Structure Review

After implementation, review:

* whether files became too large
* whether functions/classes have clear responsibilities
* whether duplicated logic should be extracted
* whether constants should be centralized
* whether tests cover changed behavior
* whether names are clear
* whether map/config validation remains explicit

Prefer small follow-up cleanup over broad rewrites.

---

## Documentation Updates

After meaningful changes, update relevant documentation:

* `README.md`
* `DEVELOPMENT_GUIDE.md`
* files under `docs/`
* map/config examples
* testing instructions
* action/observation descriptions
* render/UI behavior
* reward/game-over behavior

Documentation must match actual behavior.

---

## Worklog Requirement

After each completed task, update or create a worklog entry.

Preferred location:

```text
docs/worklog/YYYY-MM-DD.md
```

Use this format:

```md
# YYYY-MM-DD

## Summary
- ...

## Files Changed
- `path/to/file.py`: ...
- `path/to/test.py`: ...

## Behavior Changes
- ...

## Tests
- Command: `source .venv/bin/activate && python -m pytest -q`
- Result: pass/fail
- Notes: ...

## Known Issues / Next Steps
- ...
```

If a worklog file for the date already exists, append a new section instead of overwriting unrelated entries.

---

## Constraints

Do not:

* copy copyrighted assets, maps, names, sprites, music, or proprietary game data
* introduce large dependencies without approval
* rewrite the whole project unless explicitly requested
* change public environment semantics without updating tests and docs
* ignore failing smoke tests
* remove tests only to make the suite pass
* silently change action meanings or observation format

---

## Project-Specific Defaults

Unless the project documentation says otherwise:

* tile size: `16 x 16` pixels
* map area: `10 x 8` tiles
* HUD area: `10 x 2` tiles
* render size: `160 x 160` pixels
* action space:

  * `0 = no-op`
  * `1 = up`
  * `2 = down`
  * `3 = left`
  * `4 = right`
  * `5 = interact`
* no-op still advances the environment tick
* monster speed defaults to `player_speed * 0.8`
* dynamic entities use pixel-level positions
* static map layout uses tile-level coordinates
* game over occurs when player health is `<= 0`

---

## Final Response Format

After completing changes, report:

```md
## Completed

### Summary
- ...

### Tests Run
- `...` — pass/fail

### Files Changed
- ...

### Documentation Updated
- ...

### Known Issues
- ...
```
