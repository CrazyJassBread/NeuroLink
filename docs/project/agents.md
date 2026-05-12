# Repository Agent Guide

## Required Reading Order

Before making meaningful changes, read:

1. `docs/project/agents.md`
2. `docs/project/development-guide.md`
3. `README.md`
4. `docs/README.md`
5. the relevant files under `docs/guides/` and `docs/reference/`
6. relevant source files
7. relevant tests
8. relevant map/config examples

Do not modify code or docs before producing a plan and receiving approval.

## Workflow

For each development task:

1. Inspect the current implementation.
2. Identify the minimal required change set.
3. Produce a plan.
4. Wait for approval.
5. Implement incrementally.
6. Run the most relevant smoke tests using the project `.venv`.
7. Fix obvious failures introduced by the change.
8. Re-run tests.
9. Update the canonical docs if behavior or guidance changed.
10. Update the worklog.
11. Summarize results clearly.

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
```

Do not edit files until the plan is approved.

## Virtual Environment

Use the existing `.venv`.

Preferred commands:

```bash
source .venv/bin/activate
python --version
python -m pip list
```

If `.venv` is missing, report it instead of creating a new one unless the user
explicitly asks.

## Smoke Test Priority

After changes, run the most relevant existing checks first.

Typical commands:

```bash
source .venv/bin/activate
python -m pytest -q
python -m pytest tests
```

Only run lint/type tools that already exist in the environment.

## Documentation Policy

When behavior or workflow changes, update the canonical location instead of
copying the same explanation into multiple files.

Document types:

- `docs/guides/`: walkthroughs and conceptual guides
- `docs/reference/`: exact interfaces and behavior definitions
- `docs/project/`: workflow, roadmap, and contributor policy
- `docs/worklog/`: dated historical records

### Adding New Documents

When adding new documentation, prefer the current docs architecture instead of
creating new general-purpose files directly under `docs/`.

Default placement rules:

- add human-readable walkthroughs, tutorials, and usage guides under
  `docs/guides/`
- add exact schemas, API notes, field definitions, and behavior contracts under
  `docs/reference/`
- add workflow, roadmap, process, or contributor policy under `docs/project/`
- add dated historical notes only under `docs/worklog/`

Recommended root-level exceptions:

- `docs/README.md` as the documentation hub
- rare top-level files that are intentionally acting as directory navigation
  entrypoints

Before creating a new document, first check whether the information belongs in
an existing canonical file. Prefer extending or splitting existing docs over
creating overlapping explanations.

If you add, move, split, or merge a user-facing document:

- update `docs/README.md` if the navigation should change
- update README links in `README.md`, `env_diy/README.md`, or `rl/README.md`
  when their references change
- do not leave new documentation discoverable only by filename with no hub or
  README entry when it should be part of the normal documentation flow

## Worklog Policy

After a completed task, add or update:

```text
docs/worklog/YYYY-MM-DD.md
```

If a file for the day already exists, append a new dated section instead of
overwriting unrelated notes.

## Constraints

Do not:

- copy copyrighted assets or proprietary game data
- rewrite the whole project unless explicitly requested
- silently change action meanings or observation format
- remove tests just to make them pass
- leave docs in a duplicated or contradictory state after a change
