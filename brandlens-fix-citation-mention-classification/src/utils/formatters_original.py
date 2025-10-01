"""
Output formatters for BrandLens analysis results.

Provides JSON and Rich terminal formatting for analysis results,
progress indicators, and summary statistics.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.align import Align

from ..core.models import BrandAnalysis, Citation, Mention, PerformanceMetrics


class JSONFormatter:
    """JSON output formatter with pretty printing and validation."""

    @staticmethod
    def format_analysis(analysis: BrandAnalysis, indent: int = 2) -> str:
        """Format BrandAnalysis as pretty JSON."""
        return json.dumps(analysis.model_dump(mode="json"), indent=indent, ensure_ascii=False)

    @staticmethod
    def format_summary(analysis: BrandAnalysis) -> str:
        """Format analysis summary as compact JSON."""
        summary = {
            "brand": analysis.metadata.get("brand", "Unknown"),
            "visibility_score": analysis.advanced_metrics.get("visibility_score", 0),
            "citations_count": len(analysis.citations),
            "mentions_count": len(analysis.mentions),
            "cost_usd": analysis.metadata.get("cost_usd", 0),
            "processing_time_ms": analysis.metadata.get("processing_time_ms", 0),
            "timestamp": analysis.metadata.get("timestamp", datetime.now().isoformat())
        }
        return json.dumps(summary, indent=2)


class RichFormatter:
    """Rich terminal formatter for beautiful console output."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def format_analysis_overview(self, analysis: BrandAnalysis) -> Panel:
        """Create overview panel for analysis results."""
        brand = analysis.metadata.get("brand", "Unknown Brand")
        visibility_score = analysis.advanced_metrics.get("visibility_score", 0)
        cost = analysis.metadata.get("cost_usd", 0)
        processing_time = analysis.metadata.get("processing_time_ms", 0)

        # Create overview text
        overview_text = Text()
        overview_text.append(f"Brand: ", style="bold")
        overview_text.append(f"{brand}\n", style="cyan")
        overview_text.append(f"Visibility Score: ", style="bold")
        overview_text.append(f"{visibility_score:.1f}/10\n", style="green" if visibility_score >= 7 else "yellow" if visibility_score >= 4 else "red")
        overview_text.append(f"Cost: ", style="bold")
        overview_text.append(f"${cost:.4f}\n", style="green" if cost < 0.03 else "yellow")
        overview_text.append(f"Processing Time: ", style="bold")
        overview_text.append(f"{processing_time/1000:.1f}s", style="blue")

        return Panel(
            overview_text,
            title="[bold]Analysis Overview[/bold]",
            border_style="blue"
        )

    def format_citations_table(self, citations: List[Citation]) -> Table:
        """Create table for citations."""
        table = Table(title="Citations Found")
        table.add_column("Text", style="cyan", no_wrap=False, max_width=40)
        table.add_column("URL", style="blue", no_wrap=False, max_width=50)
        table.add_column("Entity", style="green")
        table.add_column("Confidence", style="yellow", justify="right")

        for citation in citations:
            confidence_color = "green" if citation.confidence >= 0.9 else "yellow" if citation.confidence >= 0.7 else "red"
            table.add_row(
                citation.text,
                citation.url,
                citation.entity,
                f"[{confidence_color}]{citation.confidence:.2f}[/{confidence_color}]"
            )

        return table

    def format_mentions_table(self, mentions: List[Mention]) -> Table:
        """Create table for mentions."""
        table = Table(title="Brand Mentions")
        table.add_column("Type", style="magenta")
        table.add_column("Text", style="cyan", no_wrap=False, max_width=30)
        table.add_column("Position", style="blue", justify="right")
        table.add_column("Context", style="white", no_wrap=False, max_width=60)

        for mention in mentions:
            type_color = "green" if mention.type == "linked" else "yellow"
            table.add_row(
                f"[{type_color}]{mention.type.title()}[/{type_color}]",
                mention.text,
                str(mention.position),
                mention.context
            )

        return table

    def format_metrics_table(self, metrics: Dict[str, Any]) -> Table:
        """Create table for advanced metrics."""
        table = Table(title="Advanced Metrics")
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="green", justify="right")
        table.add_column("Interpretation", style="white")

        # Visibility score
        visibility = metrics.get("visibility_score", 0)
        vis_interpretation = "Excellent" if visibility >= 8 else "Good" if visibility >= 6 else "Fair" if visibility >= 4 else "Poor"
        table.add_row("Visibility Score", f"{visibility:.1f}/10", vis_interpretation)

        # Share of voice
        sov = metrics.get("share_of_voice", 0)
        sov_interpretation = "Dominant" if sov >= 0.7 else "Strong" if sov >= 0.5 else "Moderate" if sov >= 0.3 else "Weak"
        table.add_row("Share of Voice", f"{sov:.1%}", sov_interpretation)

        # Position adjusted score
        pas = metrics.get("position_adjusted_score", 0)
        table.add_row("Position Score", f"{pas:.2f}", "Higher is better")

        # Competitive analysis
        competitive = metrics.get("competitive_analysis", {})
        dominance = competitive.get("brand_dominance_ratio", 0)
        dom_interpretation = "Market Leader" if dominance >= 0.8 else "Strong Player" if dominance >= 0.6 else "Competitive" if dominance >= 0.4 else "Challenger"
        table.add_row("Brand Dominance", f"{dominance:.1%}", dom_interpretation)

        return table

    def format_sources_table(self, owned_sources: List[str], external_sources: List[str]) -> Table:
        """Create table for sources breakdown."""
        table = Table(title="Sources Analysis")
        table.add_column("Type", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Examples", no_wrap=False, max_width=60)

        # Owned sources
        owned_examples = ", ".join(owned_sources[:3]) + ("..." if len(owned_sources) > 3 else "")
        table.add_row(
            "[green]Owned[/green]",
            f"[green]{len(owned_sources)}[/green]",
            owned_examples
        )

        # External sources
        external_examples = ", ".join(external_sources[:3]) + ("..." if len(external_sources) > 3 else "")
        table.add_row(
            "[blue]External[/blue]",
            f"[blue]{len(external_sources)}[/blue]",
            external_examples
        )

        return table

    def display_analysis(self, analysis: BrandAnalysis) -> None:
        """Display complete analysis with rich formatting."""
        # Overview panel
        self.console.print(self.format_analysis_overview(analysis))
        self.console.print()

        # Citations table
        if analysis.citations:
            self.console.print(self.format_citations_table(analysis.citations))
            self.console.print()

        # Mentions table
        if analysis.mentions:
            self.console.print(self.format_mentions_table(analysis.mentions))
            self.console.print()

        # Metrics table
        if analysis.advanced_metrics:
            self.console.print(self.format_metrics_table(analysis.advanced_metrics))
            self.console.print()

        # Sources table
        self.console.print(self.format_sources_table(analysis.owned_sources, analysis.sources))

        # Content gaps and insights
        if analysis.advanced_metrics.get("content_gaps"):
            gaps_text = "\n".join(f"• {gap}" for gap in analysis.advanced_metrics["content_gaps"])
            self.console.print(Panel(gaps_text, title="[yellow]Content Gaps[/yellow]", border_style="yellow"))
            self.console.print()

        if analysis.advanced_metrics.get("insights"):
            insights_text = "\n".join(f"• {insight}" for insight in analysis.advanced_metrics["insights"])
            self.console.print(Panel(insights_text, title="[green]Key Insights[/green]", border_style="green"))


class ProgressIndicator:
    """Progress indicators for long-running operations."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.progress = None

    def create_analysis_progress(self) -> Progress:
        """Create progress bar for analysis pipeline."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console
        )

    def __enter__(self):
        self.progress = self.create_analysis_progress()
        self.progress.__enter__()
        return self

    def __exit__(self, *args):
        if self.progress:
            self.progress.__exit__(*args)

    def add_task(self, description: str, total: Optional[float] = None) -> int:
        """Add a task to track."""
        if self.progress:
            return self.progress.add_task(description, total=total)
        return 0

    def update_task(self, task_id: int, advance: Optional[float] = None, **kwargs) -> None:
        """Update task progress."""
        if self.progress:
            self.progress.update(task_id, advance=advance, **kwargs)

    def complete_task(self, task_id: int) -> None:
        """Mark task as complete."""
        if self.progress:
            self.progress.update(task_id, completed=True)


class SummaryStatistics:
    """Summary statistics formatter."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def format_performance_summary(self, metrics: PerformanceMetrics) -> Panel:
        """Format performance metrics summary."""
        perf_text = Text()
        perf_text.append("API Calls:\n", style="bold")
        perf_text.append(f"  Tavily: {metrics.api_calls.get('tavily', 0)}\n")
        perf_text.append(f"  Gemini: {metrics.api_calls.get('gemini', 0)}\n")
        perf_text.append(f"Total Tokens: {metrics.total_tokens:,}\n", style="bold")
        perf_text.append(f"Cache Hit Rate: {metrics.cache_hit_rate:.1f}%\n", style="green" if metrics.cache_hit_rate >= 60 else "yellow")
        perf_text.append(f"Cache Hits: {metrics.cache_hits}\n", style="blue")
        perf_text.append(f"Total Cost: ${metrics.total_cost_usd:.4f}\n", style="green" if metrics.total_cost_usd < 0.05 else "yellow")

        return Panel(
            perf_text,
            title="[bold]Performance Summary[/bold]",
            border_style="green"
        )

    def format_cost_breakdown(self, analysis: BrandAnalysis) -> Table:
        """Create cost breakdown table."""
        table = Table(title="Cost Breakdown")
        table.add_column("Service", style="cyan")
        table.add_column("Usage", style="blue")
        table.add_column("Cost", style="green", justify="right")

        metadata = analysis.metadata
        api_calls = metadata.get("api_calls", {})

        # Tavily costs
        tavily_calls = api_calls.get("tavily", 0)
        tavily_cost = tavily_calls * 0.001
        table.add_row("Tavily Search", f"{tavily_calls} searches", f"${tavily_cost:.4f}")

        # Gemini costs
        gemini_calls = api_calls.get("gemini", 0)
        total_tokens = metadata.get("total_tokens", 0)
        input_tokens = metadata.get("prompt_tokens", 0)
        output_tokens = metadata.get("completion_tokens", 0)

        input_cost = (input_tokens / 1000) * 0.00001875
        output_cost = (output_tokens / 1000) * 0.0000375
        gemini_cost = input_cost + output_cost

        table.add_row("Gemini LLM", f"{total_tokens:,} tokens", f"${gemini_cost:.4f}")

        # Total
        total_cost = metadata.get("cost_usd", tavily_cost + gemini_cost)
        table.add_row("[bold]Total[/bold]", "", f"[bold]${total_cost:.4f}[/bold]")

        return table

    def display_summary(self, analysis: BrandAnalysis) -> None:
        """Display comprehensive summary."""
        # Performance summary
        if hasattr(analysis, "performance_metrics"):
            self.console.print(self.format_performance_summary(analysis.performance_metrics))
            self.console.print()

        # Cost breakdown
        self.console.print(self.format_cost_breakdown(analysis))


# Convenience functions
def format_json(analysis: BrandAnalysis, indent: int = 2) -> str:
    """Quick JSON formatting."""
    return JSONFormatter.format_analysis(analysis, indent)


def display_rich(analysis: BrandAnalysis, console: Optional[Console] = None) -> None:
    """Quick rich display."""
    formatter = RichFormatter(console)
    formatter.display_analysis(analysis)


def create_progress() -> ProgressIndicator:
    """Create progress indicator."""
    return ProgressIndicator()