"""
Refactored output formatters for BrandLens analysis results.

Improved version with better separation of concerns, reduced duplication,
and enhanced maintainability through design patterns.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Protocol

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


class ThresholdLevel(Enum):
    """Threshold levels for color coding."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class MetricThreshold:
    """Defines thresholds for metric evaluation."""

    def __init__(self, excellent: float, good: float, fair: float):
        self.excellent = excellent
        self.good = good
        self.fair = fair

    def evaluate(self, value: float) -> ThresholdLevel:
        """Evaluate value against thresholds."""
        try:
            # Ensure value is numeric
            numeric_value = float(value) if value is not None else 0.0

            if numeric_value >= self.excellent:
                return ThresholdLevel.EXCELLENT
            elif numeric_value >= self.good:
                return ThresholdLevel.GOOD
            elif numeric_value >= self.fair:
                return ThresholdLevel.FAIR
            else:
                return ThresholdLevel.POOR
        except (TypeError, ValueError):
            # Return POOR for non-numeric values
            return ThresholdLevel.POOR


class ColorScheme:
    """Centralized color scheme management."""

    COLORS = {
        ThresholdLevel.EXCELLENT: "green",
        ThresholdLevel.GOOD: "blue",
        ThresholdLevel.FAIR: "yellow",
        ThresholdLevel.POOR: "red"
    }

    @classmethod
    def get_color(cls, level: ThresholdLevel) -> str:
        """Get color for threshold level."""
        return cls.COLORS[level]


class MetadataExtractor:
    """Centralized metadata extraction with defaults."""

    @staticmethod
    def extract_brand(analysis: BrandAnalysis) -> str:
        """Extract brand name with fallback."""
        # Try BrandAnalysis.brand_name field first, then metadata fallback
        if hasattr(analysis, 'brand_name') and analysis.brand_name:
            return analysis.brand_name
        return analysis.metadata.get("brand", "Unknown")

    @staticmethod
    def extract_cost(analysis: BrandAnalysis) -> float:
        """Extract cost with fallback."""
        # Check nested performance metadata first
        performance = analysis.metadata.get("performance", {})
        if "total_cost_usd" in performance:
            return performance["total_cost_usd"]
        # Fallback to top-level metadata
        return analysis.metadata.get("cost_usd", 0.0)

    @staticmethod
    def extract_processing_time(analysis: BrandAnalysis) -> int:
        """Extract processing time with fallback."""
        # Check nested performance metadata first
        performance = analysis.metadata.get("performance", {})
        if "total_time_ms" in performance:
            return int(performance["total_time_ms"])
        # Fallback to top-level metadata
        return analysis.metadata.get("processing_time_ms", 0)

    @staticmethod
    def extract_api_calls(analysis: BrandAnalysis) -> Dict[str, int]:
        """Extract API calls with fallback."""
        return analysis.metadata.get("api_calls", {})

    @staticmethod
    def extract_visibility_score(analysis: BrandAnalysis) -> float:
        """Extract visibility score with fallback."""
        return analysis.advanced_metrics.get("visibility_score", 0.0)


class FormatterMixin:
    """Mixin providing common formatter functionality."""

    def __init__(self, console: Optional[Console] = None):
        self._console = console or Console()

    @property
    def console(self) -> Console:
        """Get console instance."""
        return self._console

    def create_base_table(self, title: str) -> Table:
        """Create table with consistent styling."""
        table = Table(title=title)
        return table

    def format_percentage(self, value: float) -> str:
        """Format percentage with consistent precision."""
        return f"{value:.1%}"

    def format_currency(self, value: float) -> str:
        """Format currency with consistent precision."""
        return f"${value:.4f}"

    def format_time_ms(self, value: int) -> str:
        """Format time in milliseconds to seconds."""
        return f"{value/1000:.1f}s"


class TableBuilder:
    """Builder pattern for creating Rich tables."""

    def __init__(self, title: str):
        self._table = Table(title=title)

    def add_column(self, header: str, style: str = "white",
                   justify: str = "left", no_wrap: bool = True,
                   max_width: Optional[int] = None) -> "TableBuilder":
        """Add column with fluent interface."""
        self._table.add_column(
            header,
            style=style,
            justify=justify,
            no_wrap=no_wrap,
            max_width=max_width
        )
        return self

    def add_row(self, *values: str) -> "TableBuilder":
        """Add row with fluent interface."""
        self._table.add_row(*values)
        return self

    def build(self) -> Table:
        """Build the final table."""
        return self._table


class JSONFormatter:
    """JSON output formatter with enhanced error handling."""

    class Config:
        """Configuration for JSON formatting."""
        DEFAULT_INDENT = 2
        ENSURE_ASCII = False

    @classmethod
    def format_analysis(cls, analysis: BrandAnalysis, indent: int = None) -> str:
        """Format BrandAnalysis as pretty JSON with error handling."""
        try:
            indent = indent or cls.Config.DEFAULT_INDENT
            return json.dumps(
                analysis.model_dump(mode="json"),
                indent=indent,
                ensure_ascii=cls.Config.ENSURE_ASCII
            )
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize analysis to JSON: {e}") from e

    @classmethod
    def format_summary(cls, analysis: BrandAnalysis) -> str:
        """Format analysis summary as compact JSON."""
        try:
            summary = {
                "brand": MetadataExtractor.extract_brand(analysis),
                "visibility_score": MetadataExtractor.extract_visibility_score(analysis),
                "citations_count": len(analysis.citations),
                "mentions_count": len(analysis.mentions),
                "cost_usd": MetadataExtractor.extract_cost(analysis),
                "processing_time_ms": MetadataExtractor.extract_processing_time(analysis),
                "timestamp": analysis.metadata.get("timestamp", datetime.now().isoformat())
            }
            return json.dumps(summary, indent=cls.Config.DEFAULT_INDENT)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to create summary JSON: {e}") from e


class MetricEvaluator:
    """Evaluates metrics against predefined thresholds."""

    # Define thresholds for different metrics
    VISIBILITY_THRESHOLD = MetricThreshold(8.0, 6.0, 4.0)
    SOV_THRESHOLD = MetricThreshold(0.7, 0.5, 0.3)
    DOMINANCE_THRESHOLD = MetricThreshold(0.8, 0.6, 0.4)
    COST_THRESHOLD = MetricThreshold(0.02, 0.03, 0.05)  # Inverted: lower is better
    CACHE_THRESHOLD = MetricThreshold(80.0, 60.0, 40.0)

    @classmethod
    def evaluate_visibility(cls, score: float) -> ThresholdLevel:
        """Evaluate visibility score."""
        return cls.VISIBILITY_THRESHOLD.evaluate(score)

    @classmethod
    def evaluate_sov(cls, sov: float) -> ThresholdLevel:
        """Evaluate share of voice."""
        return cls.SOV_THRESHOLD.evaluate(sov)

    @classmethod
    def evaluate_dominance(cls, dominance: float) -> ThresholdLevel:
        """Evaluate brand dominance."""
        return cls.DOMINANCE_THRESHOLD.evaluate(dominance)

    @classmethod
    def evaluate_cost(cls, cost: float) -> ThresholdLevel:
        """Evaluate cost (inverted logic - lower is better)."""
        if cost <= cls.COST_THRESHOLD.excellent:
            return ThresholdLevel.EXCELLENT
        elif cost <= cls.COST_THRESHOLD.good:
            return ThresholdLevel.GOOD
        elif cost <= cls.COST_THRESHOLD.fair:
            return ThresholdLevel.FAIR
        else:
            return ThresholdLevel.POOR

    @classmethod
    def evaluate_cache_rate(cls, rate: float) -> ThresholdLevel:
        """Evaluate cache hit rate."""
        return cls.CACHE_THRESHOLD.evaluate(rate)


class RichFormatter(FormatterMixin):
    """Rich terminal formatter with improved organization."""

    def format_analysis_overview(self, analysis: BrandAnalysis) -> Panel:
        """Create overview panel with enhanced formatting."""
        try:
            brand = MetadataExtractor.extract_brand(analysis)
            visibility_score = MetadataExtractor.extract_visibility_score(analysis)
            cost = MetadataExtractor.extract_cost(analysis)
            processing_time = MetadataExtractor.extract_processing_time(analysis)

            # Evaluate metrics
            vis_level = MetricEvaluator.evaluate_visibility(visibility_score)
            cost_level = MetricEvaluator.evaluate_cost(cost)

            # Create styled text with safe formatting
            overview_text = Text()
            overview_text.append("Brand: ", style="bold")
            overview_text.append(f"{brand}\n", style="cyan")
            overview_text.append("Visibility Score: ", style="bold")

            # Safe numeric formatting
            try:
                vis_score_str = f"{float(visibility_score):.1f}/10\n"
            except (ValueError, TypeError):
                vis_score_str = f"{visibility_score}/10\n"

            overview_text.append(
                vis_score_str,
                style=ColorScheme.get_color(vis_level)
            )
            overview_text.append("Cost: ", style="bold")
            overview_text.append(
                f"{self.format_currency(cost)}\n",
                style=ColorScheme.get_color(cost_level)
            )
            overview_text.append("Processing Time: ", style="bold")
            overview_text.append(self.format_time_ms(processing_time), style="blue")

            return Panel(
                overview_text,
                title="[bold]Analysis Overview[/bold]",
                border_style="blue"
            )
        except Exception as e:
            # Fallback panel for errors
            error_text = Text(f"Error formatting overview: {e}", style="red")
            return Panel(
                error_text,
                title="[bold]Analysis Overview (Error)[/bold]",
                border_style="red"
            )

    def format_citations_table(self, citations: List[Citation]) -> Table:
        """Create citations table with builder pattern."""
        if not citations:
            return self._create_empty_table("Citations Found", "No citations found")

        builder = (TableBuilder("Citations Found")
                  .add_column("Text", style="cyan", no_wrap=False, max_width=40)
                  .add_column("URL", style="blue", no_wrap=False, max_width=50)
                  .add_column("Entity", style="green")
                  .add_column("Confidence", style="yellow", justify="right"))

        for citation in citations:
            confidence_level = self._evaluate_confidence(citation.confidence)
            confidence_color = ColorScheme.get_color(confidence_level)

            builder.add_row(
                citation.text,
                citation.url,
                citation.entity,
                f"[{confidence_color}]{citation.confidence:.2f}[/{confidence_color}]"
            )

        return builder.build()

    def format_mentions_table(self, mentions: List[Mention]) -> Table:
        """Create mentions table with builder pattern."""
        if not mentions:
            return self._create_empty_table("Brand Mentions", "No mentions found")

        builder = (TableBuilder("Brand Mentions")
                  .add_column("Type", style="magenta")
                  .add_column("Text", style="cyan", no_wrap=False, max_width=30)
                  .add_column("Position", style="blue", justify="right")
                  .add_column("Context", style="white", no_wrap=False, max_width=60))

        for mention in mentions:
            type_color = "green" if mention.type == "linked" else "yellow"
            builder.add_row(
                f"[{type_color}]{mention.type.title()}[/{type_color}]",
                mention.text,
                str(mention.position),
                mention.context
            )

        return builder.build()

    def format_metrics_table(self, metrics: Dict[str, Any]) -> Table:
        """Create metrics table with enhanced evaluation."""
        try:
            builder = (TableBuilder("Advanced Metrics")
                      .add_column("Metric", style="bold cyan")
                      .add_column("Value", style="green", justify="right")
                      .add_column("Interpretation", style="white"))

            # Visibility score with safe formatting
            visibility = metrics.get("visibility_score", 0)
            vis_level = MetricEvaluator.evaluate_visibility(visibility)
            try:
                vis_value = f"{float(visibility):.1f}/10"
            except (ValueError, TypeError):
                vis_value = f"{visibility}/10"

            builder.add_row(
                "Visibility Score",
                vis_value,
                vis_level.value.title()
            )

            # Share of voice with safe formatting
            sov = metrics.get("share_of_voice", 0)
            sov_level = MetricEvaluator.evaluate_sov(sov)
            try:
                sov_value = self.format_percentage(float(sov))
            except (ValueError, TypeError):
                sov_value = str(sov)

            builder.add_row(
                "Share of Voice",
                sov_value,
                sov_level.value.title()
            )

            # Position adjusted score with safe formatting
            pas = metrics.get("position_adjusted_score", 0)
            try:
                pas_value = f"{float(pas):.2f}"
            except (ValueError, TypeError):
                pas_value = str(pas)

            builder.add_row("Position Score", pas_value, "Higher is better")

            # Competitive analysis with safe formatting
            competitive = metrics.get("competitive_analysis", {})
            dominance = competitive.get("brand_dominance_ratio", 0)
            dom_level = MetricEvaluator.evaluate_dominance(dominance)
            try:
                dom_value = self.format_percentage(float(dominance))
            except (ValueError, TypeError):
                dom_value = str(dominance)

            builder.add_row(
                "Brand Dominance",
                dom_value,
                dom_level.value.replace('_', ' ').title()
            )

            return builder.build()
        except Exception as e:
            # Return error table
            error_table = Table(title="Advanced Metrics (Error)")
            error_table.add_column("Error", style="red")
            error_table.add_row(f"Error formatting metrics: {e}")
            return error_table

    def format_sources_table(self, owned_sources: List[str], external_sources: List[str]) -> Table:
        """Create sources table with enhanced formatting."""
        builder = (TableBuilder("Sources Analysis")
                  .add_column("Type", style="bold")
                  .add_column("Count", justify="right")
                  .add_column("Examples", no_wrap=False, max_width=60))

        # Helper function to format examples
        def format_examples(sources: List[str], max_examples: int = 3) -> str:
            if not sources:
                return "None"
            examples = ", ".join(sources[:max_examples])
            return examples + ("..." if len(sources) > max_examples else "")

        # Owned sources
        builder.add_row(
            "[green]Owned[/green]",
            f"[green]{len(owned_sources)}[/green]",
            format_examples(owned_sources)
        )

        # External sources
        builder.add_row(
            "[blue]External[/blue]",
            f"[blue]{len(external_sources)}[/blue]",
            format_examples(external_sources)
        )

        return builder.build()

    def display_analysis(self, analysis: BrandAnalysis) -> None:
        """Display complete analysis with improved structure."""
        display_components = [
            ("overview", lambda: self.format_analysis_overview(analysis)),
            ("citations", lambda: self.format_citations_table(analysis.citations) if analysis.citations else None),
            ("mentions", lambda: self.format_mentions_table(analysis.mentions) if analysis.mentions else None),
            ("metrics", lambda: self.format_metrics_table(analysis.advanced_metrics) if analysis.advanced_metrics else None),
            ("sources", lambda: self.format_sources_table(analysis.owned_sources, analysis.sources)),
        ]

        for name, component_func in display_components:
            try:
                component = component_func()
                if component:
                    self.console.print(component)
                    self.console.print()
            except Exception as e:
                self.console.print(f"[red]Error displaying {name}: {e}[/red]")

        # Display additional insights
        self._display_insights_panels(analysis)

    def _display_insights_panels(self, analysis: BrandAnalysis) -> None:
        """Display content gaps and insights panels."""
        # Content gaps
        content_gaps = analysis.advanced_metrics.get("content_gaps", [])
        if content_gaps:
            gaps_text = "\n".join(f"• {gap}" for gap in content_gaps)
            self.console.print(
                Panel(gaps_text, title="[yellow]Content Gaps[/yellow]", border_style="yellow")
            )
            self.console.print()

        # Insights
        insights = analysis.advanced_metrics.get("insights", [])
        if insights:
            insights_text = "\n".join(f"• {insight}" for insight in insights)
            self.console.print(
                Panel(insights_text, title="[green]Key Insights[/green]", border_style="green")
            )

    def _create_empty_table(self, title: str, message: str) -> Table:
        """Create table for empty data sets."""
        table = Table(title=title)
        table.add_column("Message", style="yellow")
        table.add_row(message)
        return table

    def _evaluate_confidence(self, confidence: float) -> ThresholdLevel:
        """Evaluate confidence level."""
        if confidence >= 0.9:
            return ThresholdLevel.EXCELLENT
        elif confidence >= 0.7:
            return ThresholdLevel.GOOD
        elif confidence >= 0.5:
            return ThresholdLevel.FAIR
        else:
            return ThresholdLevel.POOR


class ProgressIndicator(FormatterMixin):
    """Enhanced progress indicator with better error handling."""

    def __init__(self, console: Optional[Console] = None):
        super().__init__(console)
        self._progress = None

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

    def __enter__(self) -> "ProgressIndicator":
        """Enter context manager."""
        try:
            self._progress = self.create_analysis_progress()
            self._progress.__enter__()
        except Exception as e:
            self.console.print(f"[red]Failed to initialize progress: {e}[/red]")
            self._progress = None
        return self

    def __exit__(self, *args) -> None:
        """Exit context manager."""
        if self._progress:
            try:
                self._progress.__exit__(*args)
            except Exception as e:
                self.console.print(f"[red]Error closing progress: {e}[/red]")
            finally:
                self._progress = None

    def add_task(self, description: str, total: Optional[float] = None) -> int:
        """Add a task to track with error handling."""
        if self._progress:
            try:
                return self._progress.add_task(description, total=total)
            except Exception as e:
                self.console.print(f"[red]Failed to add task '{description}': {e}[/red]")
        return 0

    def update_task(self, task_id: int, advance: Optional[float] = None, **kwargs) -> None:
        """Update task progress with error handling."""
        if self._progress:
            try:
                self._progress.update(task_id, advance=advance, **kwargs)
            except Exception as e:
                self.console.print(f"[red]Failed to update task {task_id}: {e}[/red]")

    def complete_task(self, task_id: int) -> None:
        """Mark task as complete with error handling."""
        if self._progress:
            try:
                self._progress.update(task_id, completed=True)
            except Exception as e:
                self.console.print(f"[red]Failed to complete task {task_id}: {e}[/red]")


class CostCalculator:
    """Centralized cost calculation logic."""

    # API pricing (per 1K tokens/calls)
    TAVILY_COST_PER_CALL = 0.001
    GEMINI_INPUT_COST_PER_1K = 0.00001875
    GEMINI_OUTPUT_COST_PER_1K = 0.0000375

    @classmethod
    def calculate_tavily_cost(cls, calls: int) -> float:
        """Calculate Tavily API cost."""
        return calls * cls.TAVILY_COST_PER_CALL

    @classmethod
    def calculate_gemini_cost(cls, input_tokens: int, output_tokens: int) -> float:
        """Calculate Gemini API cost."""
        input_cost = (input_tokens / 1000) * cls.GEMINI_INPUT_COST_PER_1K
        output_cost = (output_tokens / 1000) * cls.GEMINI_OUTPUT_COST_PER_1K
        return input_cost + output_cost


class SummaryStatistics(FormatterMixin):
    """Enhanced summary statistics formatter."""

    def format_performance_summary(self, metrics: PerformanceMetrics) -> Panel:
        """Format performance metrics with enhanced styling."""
        perf_text = Text()

        # API calls section
        perf_text.append("API Calls:\n", style="bold")
        api_calls = metrics.api_calls
        perf_text.append(f"  Tavily: {api_calls.get('tavily', 0)}\n")
        perf_text.append(f"  Gemini: {api_calls.get('gemini', 0)}\n")

        # Token usage
        perf_text.append(f"Total Tokens: {metrics.total_tokens:,}\n", style="bold")

        # Cache performance
        cache_level = MetricEvaluator.evaluate_cache_rate(metrics.cache_hit_rate)
        cache_color = ColorScheme.get_color(cache_level)
        perf_text.append(
            f"Cache Hit Rate: {metrics.cache_hit_rate:.1f}%\n",
            style=cache_color
        )
        perf_text.append(f"Cache Hits: {metrics.cache_hits}\n", style="blue")

        # Cost evaluation
        cost_level = MetricEvaluator.evaluate_cost(metrics.total_cost_usd)
        cost_color = ColorScheme.get_color(cost_level)
        perf_text.append(
            f"Total Cost: {self.format_currency(metrics.total_cost_usd)}\n",
            style=cost_color
        )

        return Panel(
            perf_text,
            title="[bold]Performance Summary[/bold]",
            border_style="green"
        )

    def format_cost_breakdown(self, analysis: BrandAnalysis) -> Table:
        """Create enhanced cost breakdown table."""
        builder = (TableBuilder("Cost Breakdown")
                  .add_column("Service", style="cyan")
                  .add_column("Usage", style="blue")
                  .add_column("Cost", style="green", justify="right"))

        metadata = analysis.metadata
        api_calls = MetadataExtractor.extract_api_calls(analysis)

        # Tavily costs
        tavily_calls = api_calls.get("tavily", 0)
        tavily_cost = CostCalculator.calculate_tavily_cost(tavily_calls)
        builder.add_row(
            "Tavily Search",
            f"{tavily_calls} searches",
            self.format_currency(tavily_cost)
        )

        # Gemini costs
        input_tokens = metadata.get("prompt_tokens", 0)
        output_tokens = metadata.get("completion_tokens", 0)
        total_tokens = metadata.get("total_tokens", input_tokens + output_tokens)

        gemini_cost = CostCalculator.calculate_gemini_cost(input_tokens, output_tokens)
        builder.add_row(
            "Gemini LLM",
            f"{total_tokens:,} tokens",
            self.format_currency(gemini_cost)
        )

        # Total
        total_cost = MetadataExtractor.extract_cost(analysis)
        if total_cost == 0:  # Fallback calculation
            total_cost = tavily_cost + gemini_cost

        builder.add_row(
            "[bold]Total[/bold]",
            "",
            f"[bold]{self.format_currency(total_cost)}[/bold]"
        )

        return builder.build()

    def display_summary(self, analysis: BrandAnalysis) -> None:
        """Display comprehensive summary with error handling."""
        try:
            # Performance summary
            if hasattr(analysis, "performance_metrics"):
                self.console.print(self.format_performance_summary(analysis.performance_metrics))
                self.console.print()
        except Exception as e:
            self.console.print(f"[red]Error displaying performance summary: {e}[/red]")

        try:
            # Cost breakdown
            self.console.print(self.format_cost_breakdown(analysis))
        except Exception as e:
            self.console.print(f"[red]Error displaying cost breakdown: {e}[/red]")


# Enhanced convenience functions with better error handling
def format_json(analysis: BrandAnalysis, indent: int = 2) -> str:
    """Quick JSON formatting with error handling."""
    try:
        return JSONFormatter.format_analysis(analysis, indent)
    except Exception as e:
        raise ValueError(f"Failed to format analysis as JSON: {e}") from e


def display_rich(analysis: BrandAnalysis, console: Optional[Console] = None) -> None:
    """Quick rich display with error handling."""
    try:
        formatter = RichFormatter(console)
        formatter.display_analysis(analysis)
    except Exception as e:
        console = console or Console()
        console.print(f"[red]Failed to display analysis: {e}[/red]")


def create_progress(console: Optional[Console] = None) -> ProgressIndicator:
    """Create progress indicator with optional console."""
    return ProgressIndicator(console)