# Codex Agent Prompts

Codex now has curated system prompts that mirror the specialist Claude agents. They live under `.codex/agents` and can be combined with any ad-hoc question you want to send to the Codex CLI.

## Available agents
- `hardcoded-data-detector` – security-grade hardcoded data audits for secrets, mock data, and placeholders.
- `code-refactoring-specialist` – incremental technical-debt reduction plans and refactoring guidance.
- `devops-engineer` – CI/CD, environment automation, infrastructure, and observability planning.
- `performance-optimizer` – profiling-led investigations with concrete optimisation recommendations.
- `system-architect` – up-front solution design, diagrams, and roadmap scaffolding.

## Quickstart
```bash
# from the repo root
bash scripts/run_codex_agent.sh system-architect "Design the ingestion pipeline for real-time brands data"
```

The helper script will:
1. Look up the matching `.codex/agents/<agent>.md` prompt.
2. Append your one-line request (optional) beneath the system instructions.
3. Launch `codex` with the repo root as the working directory.

> **Note:** Executable bits cannot be set automatically in this environment, so invoke the helper through `bash` or add execute permissions manually if you are working directly inside WSL.

## Tips
- For longer briefs, pass a quoted paragraph as the prompt argument or start an interactive Codex session and paste additional context once it launches.
- To inspect the underlying instructions before running Codex, open the corresponding file in `.codex/agents`.
- When automation requires different defaults (model, sandbox, etc.), extend the helper script or use `codex` flags such as `--model` or `--sandbox` directly.
- If you create new agent personas, copy the structure from an existing file and update this document so the team knows when to use it.
