# Agent Directory

This document tracks the specialist agents available for the BrandLens project and how they are currently being used. It mirrors the project tracker and the Codex prompt packs so the team can decide quickly which persona to engage.

## Active Assignments
- **Performance Optimizer**: completed Phase 3, Task 3.2 (chunker caching tuned) and scoping Task 3.3 content compressor tuning.
- **General-Purpose Agent**: on standby after wrapping the formatters consolidation and ready to assist with compressor experiments.

## Available Agents

### System Architect
- **Use for**: end-to-end solution design, technology stack choices, data models, multi-agent coordination.
- **Outputs**: architecture summary, diagrams, service boundaries, risk log, phased roadmap.
- **Codex prompt**: `.codex/agents/system-architect.md`
- **Status**: available for Phase 3 design reviews if needed.

### Hardcoded Data Detector
- **Use for**: security sweeps to catch secrets, mock data, environment placeholders.
- **Outputs**: markdown audit report with severity-ranked findings and remediation steps.
- **Codex prompt**: `.codex/agents/hardcoded-data-detector.md`
- **Status**: on standby; schedule before production deployments.

### Code Refactoring Specialist
- **Use for**: technical debt review, refactoring strategy, design-pattern guidance.
- **Outputs**: smell inventory, prioritized improvements, incremental plan with checkpoints.
- **Codex prompt**: `.codex/agents/code-refactoring-specialist.md`
- **Status**: available; last engagement covered the formatters refactor.

### DevOps Engineer
- **Use for**: CI/CD, infrastructure automation, observability, operational guardrails.
- **Outputs**: pipeline definitions, IaC modules, rollout plans, monitoring and alerting setup.
- **Codex prompt**: `.codex/agents/devops-engineer.md`
- **Status**: unassigned; slated for later integration work.

### Performance Optimizer
- **Use for**: profiling, query tuning, scalability planning, benchmarking.
- **Outputs**: bottleneck analysis, optimization recommendations, benchmark and monitoring plan.
- **Codex prompt**: `.codex/agents/performance-optimizer.md`
- **Status**: active on Phase 3 semantic chunker performance prep.

### General-Purpose Agent (Claude)
- **Use for**: quick helpers, glue work, or exploratory tasks without a dedicated persona.
- **Outputs**: varies by request; note session outcomes in the tracker when used.
- **Status**: standing by for follow-up test or documentation support.

## Usage Notes
- Reference `PROJECT_TRACKER.md` for real-time phase progress and sub-agent assignments.
- Launch Codex personas through `bash scripts/run_codex_agent.sh <agent> "<request>"` from PowerShell (use WSL or Git Bash if available).
- After each session, update this file and the tracker with the agent, task, duration, and outcome to keep records aligned.

