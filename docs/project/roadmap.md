# Project Roadmap

This document tracks the project direction at a higher level than the worklog.

## Vision

NesyLink aims to become an original 2D reinforcement learning environment and
benchmark focused on:

- long-horizon skill composition
- compositional tasks
- procedural generalization
- neuro-symbolic constraints and task structure

## Current State

The repository currently has:

- a working `env_diy` Gymnasium environment
- task-specific challenge rooms
- benchmark v0 smoke evaluation
- RL smoke scripts and PPO integration
- DreamerV3 integration for training and interface verification

## Near-Term Priorities

- [ ] keep the environment API, rewards, and task validators stable
- [ ] improve benchmark v0 metadata and evaluation outputs
- [ ] clarify benchmark documentation and reporting boundaries
- [ ] expand task coverage without breaking current determinism guarantees

## Mid-Term Priorities

- [ ] define fixed train, validation, test, and OOD split policy
- [ ] add richer benchmark metrics and stable run artifacts
- [ ] add at least one official benchmark baseline beyond random smoke results
- [ ] improve task suite organization by difficulty and behavior type

## Long-Term Priorities

- [ ] freeze an official reproducible benchmark protocol
- [ ] publish baseline reporting rules and reproducibility instructions
- [ ] extend the benchmark with stronger compositional and neuro-symbolic tasks
