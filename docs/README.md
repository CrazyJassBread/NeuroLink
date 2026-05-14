# Documentation Hub

This directory is organized by document purpose instead of keeping every file
flat at the top level.

## Start Here

- `guides/env-overview.md`
  - Human-readable overview of `env_diy`, gameplay mechanics, pure-map schema,
    and reward-module-based environment construction.
- `reference/env-api.md`
  - Canonical API reference for creating environments with `map_id/map_path`
    plus `reward_id/reward_module`, and reading observation/info fields.
- `guides/dreamer-training.md`
  - DreamerV3 integration guide for the nested base info contract.
- `guides/rl-training.md`
  - RL-focused guide for direct `map + reward` training flows.

## Guides

Guides explain how to understand or use the system at a higher level.

- `guides/env-overview.md`
  - Main environment guide for mechanics, rooms, exits, entities, pure-map
    files, generic info/events, and reward modules.
- `guides/dreamer-training.md`
  - DreamerV3 adapter guide for current nested info and reward metadata.
- `guides/rl-training.md`
  - RL training guide for Gymnasium/SB3 usage, map selection, reward selection,
    and evaluation workflow.

## Reference

Reference docs define interfaces, schemas, and precise behavior.

- `reference/env-api.md`
  - Gymnasium entrypoints, action/observation spaces, determinism, and info
    schema.
- `reference/rewards.md`
  - `BaseReward`, builtin rewards, reward module contract, and extension rules.
- `reference/tasks-and-validators.md`
  - Task API deprecation notice and replacement environment construction flow.
- `reference/benchmark-v0.md`
  - Current benchmark v0 scope, reward-driven task wiring, registry API, and random eval.

## Project

Project docs describe development workflow, collaboration, and roadmap.

- `project/development-guide.md`
  - Canonical development guide and benchmark maturity policy.
- `project/agents.md`
  - Repository-specific collaboration and implementation workflow notes.
- `project/roadmap.md`
  - Project goals and medium-term roadmap.

## History

- `worklog/`
  - Dated change logs and implementation notes.

## Documentation Boundaries

Use these rules when updating docs:

- Put usage walkthroughs and conceptual explanations in `guides/`.
- Put exact interfaces, field definitions, and machine-facing behavior in
  `reference/`.
- Put workflow, process, roadmap, and contributor-facing rules in `project/`.
- Put dated historical records only in `worklog/`.

Avoid duplicating the same detailed explanation across multiple documents.
When a README needs detail, keep it short and link to the canonical document
here.
