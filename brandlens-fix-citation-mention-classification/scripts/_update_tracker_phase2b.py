from pathlib import Path

path = Path("PROJECT_TRACKER.md")
text = path.read_text(encoding="utf-8")
text = text.replace("**Overall Progress**: 45% complete", "**Overall Progress**: 50% complete")
text = text.replace(
    "**Next Steps**:\n- Wire strategies into Tavily orchestration layer\n- Add strategy selection unit tests (pytest tests/unit/search/test_strategies.py)\n- Prepare optional integration smoke test scaffolding",
    "**Next Steps**:\n- Record Tavily usage metrics (cost, rate limits) in orchestrator\n- Prepare optional integration smoke test scaffolding\n- Plan performance-optimizer review once real API tests pass",
)
text = text.replace(
    "| 2.1 | Tavily Client Implementation | general-purpose | IN PROGRESS | IN PROGRESS | - |",
    "| 2.1 | Tavily Client Implementation | general-purpose | 2.5h | IN PROGRESS | - |",
)
path.write_text(text, encoding="utf-8")
