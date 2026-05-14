# DreamerV3 Guide

DreamerV3 now consumes the same base env API and nested `info` structure as the rest of the repo.

Current adapter assumptions:

- nested `info["env"]`
- nested `info["agent"]`
- nested `info["inventory"]`
- event counters under `info["events"]["counts"]`
- reward metadata under `info["reward"]`

The adapter no longer depends on task-specific wrapper fields.
