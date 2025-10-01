from pathlib import Path

path = Path("docs/phase2_tavily_plan.md")
text = path.read_text(encoding="utf-8")
needle = "- src/search/tavily_client.py\n  - Async client wrapper (httpx.AsyncClient)"
if "src/search/orchestrator.py" not in text and needle in text:
    text = text.replace(
        needle,
        "- src/search/orchestrator.py\n  - Strategy-driven orchestration using Tavily client\n  - Aggregates strategy responses and logs metrics\n- src/search/tavily_client.py\n  - Async client wrapper (httpx.AsyncClient)",
        1,
    )
path.write_text(text, encoding="utf-8")
