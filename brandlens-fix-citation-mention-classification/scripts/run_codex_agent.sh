#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_codex_agent.sh <agent> [prompt...]

Examples:
  bash scripts/run_codex_agent.sh system-architect "Design the messaging service"
  bash scripts/run_codex_agent.sh performance-optimizer "Review slow dashboard query"
USAGE
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "error: codex CLI not found on PATH" >&2
  exit 127
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
AGENT=$1
shift || true
PROMPT_FILE="${REPO_ROOT}/.codex/agents/${AGENT}.md"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "error: unknown agent '${AGENT}'. Available agents:" >&2
  shopt -s nullglob
  for file in "${REPO_ROOT}/.codex/agents"/*.md; do
    base=$(basename "${file}")
    printf '  - %s\n' "${base%.md}" >&2
  done
  shopt -u nullglob
  exit 1
fi

PROMPT_CONTENT=$(cat "${PROMPT_FILE}")

if [[ $# -gt 0 ]]; then
  USER_PROMPT="$*"
  PROMPT_CONTENT+=$'\n\nUser request:\n'
  PROMPT_CONTENT+="${USER_PROMPT}"
fi

exec codex --cd "${REPO_ROOT}" "${PROMPT_CONTENT}"
