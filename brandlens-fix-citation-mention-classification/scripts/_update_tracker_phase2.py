from pathlib import Path

path = Path("PROJECT_TRACKER.md")
text = path.read_text(encoding="utf-8")
text = text.replace("**Overall Progress**: 35% complete", "**Overall Progress**: 40% complete")
text = text.replace("| 1.4 | CLI Refactor Review | code-refactoring-specialist | 0.3h | COMPLETE | 9/10 |\n\n## ", "| 1.4 | CLI Refactor Review | code-refactoring-specialist | 0.3h | COMPLETE | 9/10 |\n| 2.1 | Tavily Client Implementation | general-purpose | IN PROGRESS | IN PROGRESS | - |\n\n## ")
text = text.replace("**Next Steps**:\n- Assign upcoming Phase 2 search integration tasks to sub-agents\n- Prepare Tavily client scaffolding plan", "**Next Steps**:\n- Implement Tavily client module and supporting models\n- Add Tavily client unit tests (run: pytest tests/unit/search/test_tavily_client.py)\n- Prepare optional integration smoke test scaffolding")
path.write_text(text, encoding="utf-8")
