# Documentation Hub

This directory is organized by document purpose instead of keeping every file
flat at the top level.

## Start Here

- `guides/env-overview.md`
  - Human-readable overview of `env_diy`, gameplay mechanics, map structure,
    observations, rewards, and practical usage notes.
- `reference/env-api.md`
  - Canonical API reference for creating environments, calling `reset/step`,
    and reading observation/info fields.
- `guides/training.md`
  - Training-focused guide for DreamerV3 and related training workflows.

## Guides

Guides explain how to understand or use the system at a higher level.

- `guides/env-overview.md`
  - Main environment guide for mechanics, rooms, exits, entities, rewards,
    rendering, and examples.
- `guides/training.md`
  - Training and experiment guide for DreamerV3 and RL integration notes.

## Reference

Reference docs define interfaces, schemas, and precise behavior.

- `reference/env-api.md`
  - Gymnasium entrypoints, action/observation spaces, determinism, and info
    schema.
- `reference/rewards.md`
  - Reward implementation path, reward modes, and extension rules.
- `reference/tasks-and-validators.md`
  - Task metadata flow, validator behavior, and how to add new task maps.
- `reference/benchmark-v0.md`
  - Current benchmark v0 scope, suite contents, registry API, and random eval.

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
