"""Command line interface for BrandLens."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .search.budget import BudgetLimits, BudgetManager
from .config import load_app_config
from .core import AppConfig, ConfigurationError, TavilyAPIError
from .search.orchestrator import SearchOrchestrator, SearchRunSummary
from .search.strategies import (
    BrandFocusedStrategy,
    ComparativeSearchStrategy,
    ExploratorySearchStrategy,
    FactualSearchStrategy,
    SearchStrategyContext,
)
from .search.analytics import AnalyticsReport, summarize_strategy_results
from .search.tavily_client import TavilyClient
from .utils import configure_logging, get_logger
from .analyzer import BrandAnalyzer
from .core.models import ModelName

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
console = Console()


def _resolve_env_file(config_file: Optional[str]) -> Optional[str]:
    if not config_file:
        return None
    path = Path(config_file)
    if path.is_dir():
        raise click.ClickException("Configuration file path must point to a file, not a directory.")
    return str(path)


def _get_logger(ctx: click.Context) -> logging.Logger:
    if ctx.obj is None:
        ctx.obj = {}
    logger = ctx.obj.get("logger")
    if logger is None:
        logger = get_logger(__name__)
        ctx.obj["logger"] = logger
    return logger


def _load_config(ctx: click.Context) -> AppConfig:
    if ctx.obj is None:
        ctx.obj = {}
    if "config" not in ctx.obj:
        env_file = ctx.obj.get("config_file")
        ctx.obj["config"] = load_app_config(env_file=env_file)
    return ctx.obj["config"]


def _build_default_strategies() -> list:
    # Simplified to use only Factual strategy
    # Other strategies (Comparative, Exploratory, BrandFocused) disabled for now
    return [
        FactualSearchStrategy(),
    ]


async def _run_search(
    config: AppConfig,
    query: str,
    brand_name: Optional[str],
    brand_domain: Optional[str],
    limits: BudgetLimits,
    enable_cache: bool,
) -> tuple[SearchRunSummary, AnalyticsReport]:
    context = SearchStrategyContext(query=query, brand_name=brand_name, brand_domain=brand_domain)
    budget = BudgetManager(limits)
    async with TavilyClient.from_config(config.api, enable_cache=enable_cache) as client:
        orchestrator = SearchOrchestrator(
            client,
            strategies=_build_default_strategies(),
            budget_manager=budget,
        )
        summary = await orchestrator.run(context, include_classifier=True)
    report = summarize_strategy_results(summary)
    return summary, report


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--config-file",
    type=click.Path(path_type=str, dir_okay=False, resolve_path=True),
    default=None,
    help="Optional path to a .env file that should be loaded before running commands.",
)
@click.option(
    "--log-level",
    type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Log level to use for this invocation.",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=str, dir_okay=False),
    default=None,
    help="Write logs to this file in addition to stdout.",
)
@click.version_option(__version__, prog_name="BrandLens")
@click.pass_context
def cli(ctx: click.Context, config_file: Optional[str], log_level: str, log_file: Optional[str]) -> None:
    """BrandLens command-line interface."""

    ctx.ensure_object(dict)
    ctx.obj["config_file"] = _resolve_env_file(config_file)

    configure_logging(level=log_level, log_file=log_file, force=True)
    _get_logger(ctx).debug("Starting BrandLens CLI", extra={"config_file": ctx.obj["config_file"]})


@cli.command(name="validate-config")
@click.pass_context
def validate_config(ctx: click.Context) -> None:
    """Validate BrandLens environment configuration."""

    logger = _get_logger(ctx)
    try:
        config = _load_config(ctx)
    except ConfigurationError as exc:
        logger.error("Configuration validation failed", exc_info=exc)
        raise click.ClickException(f"Configuration error: {exc}") from exc

    summary = (
        f"Gemini model: [bold]{config.api.gemini_model}[/bold]\n"
        f"Tavily search depth: [bold]{config.api.tavily_search_depth}[/bold]\n"
        f"Cache directory: [bold]{config.cache.cache_dir}[/bold]"
    )

    console.print(Panel.fit("Configuration validated successfully", border_style="green"))
    console.print(summary)
    logger.info("Configuration validation succeeded")


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Display active configuration summary (without secrets)."""

    logger = _get_logger(ctx)
    try:
        config = _load_config(ctx)
    except ConfigurationError as exc:
        logger.error("Configuration inspection failed", exc_info=exc)
        raise click.ClickException(f"Unable to load configuration: {exc}") from exc

    console.print(
        Panel(
            f"Gemini model: [cyan]{config.api.gemini_model}[/cyan]\n"
            f"Max tokens: [cyan]{config.api.gemini_max_tokens}[/cyan]\n"
            f"Search depth: [cyan]{config.api.tavily_search_depth}[/cyan]\n"
            f"Cache dir: [cyan]{config.cache.cache_dir}[/cyan]\n"
            f"Compression target tokens: [cyan]{config.compression_target_tokens}[/cyan]",
            title="BrandLens Configuration",
        )
    )
    logger.info("Displayed configuration summary")


@cli.command(name="search")
@click.argument("query")
@click.option("--brand-name", default=None, help="Optional brand name context for strategies.")
@click.option("--brand-domain", default=None, help="Optional brand domain context (e.g. brand.com).")
@click.option("--max-searches", type=int, default=5, show_default=True, help="Maximum Tavily searches for this run.")
@click.option("--max-cost", type=float, default=1.0, show_default=True, help="Maximum Tavily spend (USD) for this run.")
@click.option("--enable-cache/--disable-cache", default=False, show_default=True, help="Reuse identical Tavily responses within the run.")
@click.pass_context
def search_command(
    ctx: click.Context,
    query: str,
    brand_name: Optional[str],
    brand_domain: Optional[str],
    max_searches: int,
    max_cost: float,
    enable_cache: bool,
) -> None:
    """Execute a BrandLens search run and display aggregated metrics."""

    logger = _get_logger(ctx)

    if max_searches <= 0:
        raise click.BadParameter("--max-searches must be greater than 0")
    if max_cost <= 0:
        raise click.BadParameter("--max-cost must be greater than 0")

    config = _load_config(ctx)
    limits = BudgetLimits(max_searches=max_searches, max_cost_usd=max_cost)

    try:
        summary, report = asyncio.run(
            _run_search(
                config,
                query=query,
                brand_name=brand_name,
                brand_domain=brand_domain,
                limits=limits,
                enable_cache=enable_cache,
            )
        )
    except TavilyAPIError as exc:
        logger.error("Tavily API error during search", exc_info=exc)
        raise click.ClickException(f"Tavily API error: {exc}") from exc

    table = Table(title="Strategy Metrics")
    table.add_column("Strategy")
    table.add_column("Results", justify="right")
    table.add_column("Domains", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Avg/Call", justify="right")

    for metric in sorted(report.metrics, key=lambda m: m.strategy):
        table.add_row(
            metric.strategy,
            str(metric.total_results),
            str(metric.unique_domains),
            f"{metric.total_cost_usd:.4f}",
            f"{metric.average_results_per_call:.2f}",
        )

    console.print(table)
    console.print(
        f"Total cost: ${report.total_cost_usd:.4f} across {report.api_calls} Tavily calls.",
        style="bold",
    )

    if report.budget_state is not None:
        remaining_cost = max(limits.max_cost_usd - report.budget_state.total_cost_usd, 0.0)
        remaining_searches = max(limits.max_searches - report.budget_state.total_searches, 0)
        console.print(
            f"Budget remaining: ${remaining_cost:.4f} ({remaining_searches} searches remaining)",
            style="cyan",
        )

    if report.last_rate_limit_remaining is not None:
        console.print(f"Rate limit remaining: {report.last_rate_limit_remaining}")

    logger.info(
        "Search completed",
        extra={
            "query": query,
            "total_cost_usd": report.total_cost_usd,
            "api_calls": report.api_calls,
        },
    )


@cli.command(name="analyze")
@click.argument("brand_name")
@click.argument("query")
@click.option("--url", help="Optional brand domain URL (e.g., apple.com)")
@click.option("--competitors", help="Comma-separated list of competitor names")
@click.option("--max-searches", type=int, default=10, show_default=True, help="Maximum Tavily searches")
@click.option("--max-sources", type=int, default=5, show_default=True, help="Maximum sources to return")
@click.option("--max-cost", type=float, default=0.5, show_default=True, help="Maximum cost (USD)")
@click.option("--enable-cache/--disable-cache", default=True, show_default=True, help="Enable caching")
@click.option("--enable-compression/--disable-compression", default=True, show_default=True, help="Enable content compression")
@click.option("--compression-ratio", type=float, default=None, help="Target compression ratio (default: from .env TOKEN_COMPRESSION_TARGET)")
@click.option("--model", type=click.Choice(["flash", "pro"], case_sensitive=False), default="flash", help="Gemini model to use")
@click.option("--format", type=click.Choice(["rich", "json"], case_sensitive=False), default="rich", help="Output format")
@click.pass_context
def analyze_command(
    ctx: click.Context,
    brand_name: str,
    query: str,
    url: Optional[str],
    competitors: Optional[str],
    max_searches: int,
    max_sources: int,
    max_cost: float,
    enable_cache: bool,
    enable_compression: bool,
    compression_ratio: float,
    model: str,
    format: str,
) -> None:
    """Perform comprehensive brand visibility analysis."""

    logger = _get_logger(ctx)

    # Validate parameters
    if max_searches <= 0:
        raise click.BadParameter("--max-searches must be greater than 0")
    if max_sources <= 0:
        raise click.BadParameter("--max-sources must be greater than 0")
    if max_cost <= 0:
        raise click.BadParameter("--max-cost must be greater than 0")

    # Handle optional URL parameter - use brand name as fallback for domain
    brand_domain = url if url else f"{brand_name.lower().replace(' ', '')}.com"

    # Parse competitors
    competitor_names = None
    if competitors:
        competitor_names = [name.strip() for name in competitors.split(",") if name.strip()]

    config = _load_config(ctx)

    # Use model from config (reads from .env)
    model_name = config.api.gemini_model

    # Use compression ratio from config if not provided via CLI
    actual_compression_ratio = compression_ratio if compression_ratio is not None else config.target_compression_ratio

    # Validate compression ratio
    if not 0.1 <= actual_compression_ratio <= 0.9:
        raise click.BadParameter("compression ratio must be between 0.1 and 0.9")

    # Initialize budget limits
    budget_limits = BudgetLimits(max_searches=max_searches, max_cost_usd=max_cost)

    try:
        # Initialize analyzer
        analyzer = BrandAnalyzer(
            gemini_api_key=config.api.gemini_api_key,
            tavily_api_key=config.api.tavily_api_key,
            enable_compression=enable_compression,
            target_compression_ratio=actual_compression_ratio,
            model=model_name,
            content_mode=config.api.tavily_content_mode
        )

        # Show analysis start message only for non-JSON formats
        if format != "json":
            console.print(Panel(
                f"[bold]Starting Brand Visibility Analysis[/bold]\n\n"
                f"Brand: [cyan]{brand_name}[/cyan] ({brand_domain})\n"
                f"Query: [yellow]{query}[/yellow]\n"
                f"Competitors: [magenta]{competitors or 'None'}[/magenta]\n"
                f"Model: [blue]{model_name}[/blue]\n"
                f"Max Sources: [blue]{max_sources}[/blue]\n"
                f"Compression: [green]{enable_compression}[/green] "
                f"({actual_compression_ratio:.0%} target)" if enable_compression else f"[red]{enable_compression}[/red]",
                title="Analysis Configuration",
                border_style="blue"
            ))

        # Run analysis with status indicator only for non-JSON formats
        if format != "json":
            with console.status("[bold green]Analyzing brand visibility..."):
                analysis = asyncio.run(
                    analyzer.analyze_brand_visibility(
                        brand_name=brand_name,
                        brand_domain=brand_domain,
                        query=query,
                        competitor_names=competitor_names,
                        budget_limits=budget_limits,
                        enable_cache=enable_cache,
                        max_sources=max_sources
                    )
                )
        else:
            # Run without status indicator for JSON output
            analysis = asyncio.run(
                analyzer.analyze_brand_visibility(
                    brand_name=brand_name,
                    brand_domain=brand_domain,
                    query=query,
                    competitor_names=competitor_names,
                    budget_limits=budget_limits,
                    enable_cache=enable_cache,
                    max_sources=max_sources
                )
            )

        # Display results completion message only for non-JSON formats
        if format != "json":
            console.print("\n[bold green]✓ Analysis Complete![/bold green]\n")

        if format == "json":
            # Output the exact format required by specification
            import json
            import re

            # Strip References section from markdown for JSON output
            markdown_without_refs = re.sub(
                r'\n\n## References\n.*',
                '',
                analysis.human_response_markdown,
                flags=re.DOTALL
            ).strip()

            # Sort citations by citation number [1], [2], [3], etc.
            sorted_citations = sorted(
                analysis.citations,
                key=lambda c: int(c.text.strip('[]'))
            )

            output = {
                "human_response_markdown": markdown_without_refs,
                "citations": [
                    {
                        "text": c.text,
                        "url": c.url,
                        "entities": [c.entity]  # Convert single entity to list for consistency
                    } for c in sorted_citations
                ],
                "mentions": [
                    {
                        "text": m.text,
                        "type": m.type.value,
                        "context": m.context
                    } for m in analysis.mentions
                ],
                "owned_sources": analysis.owned_sources,
                "sources": analysis.sources,
                "metadata": {
                    "performance": analysis.metadata.get("performance", {}),
                    "search": analysis.metadata.get("search", {}),
                    "compression": analysis.metadata.get("compression", {}),
                    "llm": analysis.metadata.get("llm", {})
                }
            }
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            analyzer.display_analysis(analysis, format_type=format)

        logger.info(
            "Brand analysis completed successfully",
            extra={
                "brand_name": brand_name,
                "query": query,
                "total_cost": analysis.metadata.get("performance", {}).get("total_cost_usd", 0.0),
                "processing_time_ms": analysis.metadata.get("performance", {}).get("total_time_ms", 0.0),
                "citations": len(analysis.citations),
                "mentions": len(analysis.mentions)
            }
        )

    except Exception as exc:
        logger.error("Brand analysis failed", exc_info=exc)
        error_msg = f"Analysis failed: {exc}"
        console.print(f"[red]✗ {error_msg}[/red]")
        raise click.ClickException(error_msg) from exc
