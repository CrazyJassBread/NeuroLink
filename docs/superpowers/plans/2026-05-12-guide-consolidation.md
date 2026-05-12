# Development Guide Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the old and new development guides into one canonical document that matches the current `env_diy` and benchmark maturity, then remove the obsolete duplicate guide.

**Architecture:** Keep `docs/DEVELOPMENT_GUIDE.md` as the single canonical entrypoint. Rewrite it with the stronger benchmark framing from `DEVELOPMENT_GUIDE_NEW.md`, retain the practical environment constraints from the old guide, and explicitly stage benchmark expectations by release phase so the guide matches the current repository reality.

**Tech Stack:** Markdown, existing repository docs, `apply_patch`

---

### Task 1: Inventory guide content and references

**Files:**
- Modify: `docs/superpowers/plans/2026-05-12-guide-consolidation.md`
- Inspect: `docs/DEVELOPMENT_GUIDE.md`
- Inspect: `docs/DEVELOPMENT_GUIDE_NEW.md`
- Inspect: `readme.md`
- Inspect: `env_diy/README.md`
- Inspect: `docs/AGENTS.md`

- [ ] **Step 1: Confirm the current guide references and duplicate files**

Run: `rg -n "DEVELOPMENT_GUIDE(_NEW)?\\.md" docs readme.md env_diy/README.md`
Expected: references point to `docs/DEVELOPMENT_GUIDE.md`, with `_NEW` appearing only as the temporary duplicate guide.

- [ ] **Step 2: Identify the canonical destination**

Decision: keep `docs/DEVELOPMENT_GUIDE.md` as the final path and delete `docs/DEVELOPMENT_GUIDE_NEW.md` after its content is merged.

### Task 2: Rewrite the canonical guide

**Files:**
- Modify: `docs/DEVELOPMENT_GUIDE.md`

- [ ] **Step 1: Replace the current guide with a consolidated structure**

The rewritten guide should include:

```text
- project overview and originality constraints
- current repository reality and canonical env entrypoints
- Gymnasium / determinism / action / observation / reward / info policies
- benchmark architecture and staged maturity model
- benchmark v0, v0.1, v0.2 release gates
- workflow, plan template, testing, docs, versioning, dependency, worklog rules
```

- [ ] **Step 2: Remove duplicated or contradictory content**

The rewrite must eliminate:

```text
- duplicate numbering and section drift
- conflicting repo-structure advice that implies an immediate migration
- release requirements that are far beyond the current benchmark minimum bar
- repeated policy statements spread across multiple sections
```

### Task 3: Remove the duplicate guide and update follow-on docs if needed

**Files:**
- Delete: `docs/DEVELOPMENT_GUIDE_NEW.md`
- Modify: `docs/worklog/2026-05-12.md`

- [ ] **Step 1: Delete the obsolete duplicate**

Delete `docs/DEVELOPMENT_GUIDE_NEW.md` after the merged content is safely present in `docs/DEVELOPMENT_GUIDE.md`.

- [ ] **Step 2: Record the doc consolidation in the worklog**

Add a worklog entry summarizing:

```text
- the guide consolidation
- the new single-file canonical path
- the main benchmark policy changes
- the validation commands used
```

### Task 4: Validate the result

**Files:**
- Inspect: `docs/DEVELOPMENT_GUIDE.md`
- Inspect: `docs/worklog/2026-05-12.md`

- [ ] **Step 1: Re-scan for stale references**

Run: `rg -n "DEVELOPMENT_GUIDE_NEW\\.md" .`
Expected: no remaining references.

- [ ] **Step 2: Re-scan canonical references**

Run: `rg -n "DEVELOPMENT_GUIDE\\.md" docs readme.md env_diy/README.md`
Expected: surviving references point to the canonical file.

- [ ] **Step 3: Review git status**

Run: `git status --short`
Expected: the canonical guide is modified, the duplicate guide is deleted, and any worklog/plan additions are visible.
