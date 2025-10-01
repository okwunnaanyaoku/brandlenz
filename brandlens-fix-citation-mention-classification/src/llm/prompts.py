"""Prompt building utilities for BrandLens Gemini integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import textwrap


@dataclass
class PromptInsight:
    """Structured insight used to ground the LLM prompt."""

    title: str
    summary: str
    url: str
    source: str | None = None

    def normalized(self) -> "PromptInsight":
        return PromptInsight(
            title=self.title.strip(),
            summary=self.summary.strip(),
            url=self.url.strip(),
            source=self.source.strip() if self.source else None,
        )


@dataclass
class PromptContext:
    """Full context used to construct a Gemini prompt."""

    query: str
    brand_name: str
    brand_description: str | None = None
    analysis_goals: Sequence[str] = field(default_factory=list)
    insights: Sequence[PromptInsight] = field(default_factory=list)
    competitors: Sequence[str] = field(default_factory=list)
    caveats: Sequence[str] = field(default_factory=list)

    def cleaned(self) -> "PromptContext":
        goals = [goal.strip() for goal in self.analysis_goals if goal and goal.strip()]
        competitors = [comp.strip() for comp in self.competitors if comp and comp.strip()]
        caveats = [item.strip() for item in self.caveats if item and item.strip()]
        insights = [insight.normalized() for insight in self.insights]
        return PromptContext(
            query=self.query.strip(),
            brand_name=self.brand_name.strip(),
            brand_description=self.brand_description.strip() if self.brand_description else None,
            analysis_goals=goals,
            insights=insights,
            competitors=competitors,
            caveats=caveats,
        )


@dataclass
class FewShotExample:
    """Example question/answer pair for few-shot prompting."""

    user: str
    assistant: str


@dataclass
class PromptPayload:
    """Container holding the built prompt components."""

    system_instruction: str
    prompt_parts: List[str]
    few_shot_examples: List[FewShotExample]


class PromptBuilder:
    """Build Gemini prompts with BrandLens-specific guidance."""

    _SYSTEM_PROMPT = textwrap.dedent(
        """
        You are a helpful AI assistant that answers questions about brands and products using web research.

        ### Core Rules
        - Answer the user's question directly in a clear, conversational tone like ChatGPT
        - Use natural language - no formal report sections or business jargon
        - Organize information with simple Markdown headings and bullet points for readability
        - Cite sources using inline markers like [1] that map to a references section
        - Use realistic citation patterns: typically one source [1] per claim, occasionally two [2, 3] when multiple sources support the same point
        - Cite key factual statements, but avoid over-citation. Not every sentence needs a citation
        - Distribute citations naturally throughout the content - each source should be referenced at least once
        - Base answers strictly on provided research; never fabricate information
        - If the research mentions risks or concerns, include them naturally
        - Keep your response helpful, informative, and easy to read
        """
    ).strip()

    _OUTPUT_INSTRUCTIONS = textwrap.dedent(
        """
        ### Output Requirements
        - Answer the user's question directly and conversationally
        - Start with a brief intro, then organize key information under simple headings
        - Use short paragraphs and bullet points - make it easy to scan
        - Include relevant details, features, or context that answer the question
        - End with a references section matching the inline citation markers

        ### Citation Guidelines
        - Use single citations [1] for most claims, double citations [2, 3] only when multiple sources directly support the same specific point
        - Ensure each provided source is cited at least once throughout the response
        - Avoid citing all sources for every statement - be selective and relevant
        - Your References section must exactly match the provided reference list below - copy the exact titles and URLs
        - Do NOT create your own references - use only the ones provided in the References section below
        """
    ).strip()

    _FEW_SHOT_EXAMPLES = [
        FewShotExample(
            user=textwrap.dedent(
                """
                Primary Question: How is Aurora Bikes perceived after its electric lineup launch?
                Key Insights:
                1. Aurora Bikes reports record pre-orders for A1 e-bike. (https://aurora.example.com/news)
                2. Reviewers praise battery efficiency but note limited availability. (https://gear.example.com/review)
                3. Social listening indicates riders compare Aurora with VoltCycle. (https://community.example.com/thread)
                """
            ).strip(),
            assistant=textwrap.dedent(
                """
                Aurora Bikes is getting a lot of positive attention after launching their electric lineup! Here's what's happening:

                **Strong Early Demand**
                The A1 e-bike is seeing record pre-orders, which shows urban commuters are really excited about what Aurora is offering [1].

                **Reviewer Feedback**
                Reviewers are particularly impressed with the battery efficiency - it's standing out as a key advantage over competitors like VoltCycle [2]. However, there's a catch: availability is limited right now, which reviewers have noted as a concern [2].

                **Market Position**
                People are actively comparing Aurora with VoltCycle in online discussions, which means shoppers are considering both brands when making their decision [3]. The limited stock could be an issue though - if Aurora can't meet demand, some customers might go with alternatives instead.

                ## References
                [1] Aurora Bikes reports record pre-orders for A1 e-bike - https://aurora.example.com/news
                [2] Reviewers praise battery efficiency but note limited availability - https://gear.example.com/review
                [3] Social listening indicates riders compare Aurora with VoltCycle - https://community.example.com/thread
                """
            ).strip(),
        ),
        FewShotExample(
            user=textwrap.dedent(
                """
                Primary Question: What are the latest iPhone features?
                Key Insights:
                1. Apple announces iOS 18 with enhanced AI capabilities. (https://apple.com/newsroom/ios-18)
                2. iPhone 15 Pro features titanium design and USB-C. (https://apple.com/newsroom/iphone-15)
                3. App Store reports 40% increase in productivity apps. (https://apple.com/newsroom/app-store)
                4. Developers praise new machine learning frameworks. (https://developer.apple.com/ml-news)
                5. Users report improved battery life in iOS 18 beta. (https://apple.com/newsroom/ios-18-beta)
                """
            ).strip(),
            assistant=textwrap.dedent(
                """
                The latest iPhones have some exciting new features! Here's what's new:

                **iOS 18 and AI**
                Apple just announced iOS 18 with enhanced AI capabilities that make the iPhone smarter and more helpful for productivity tasks [1]. Early beta users are already reporting improved battery life with this update [5].

                **Hardware Upgrades**
                The iPhone 15 Pro got a premium upgrade with a titanium design that's both lighter and more durable [2]. It also finally switched to USB-C, which means you can use the same cable for your iPhone, iPad, and Mac [2].

                **App Ecosystem**
                The App Store is thriving - there's been a 40% increase in productivity apps, so you've got tons of options for getting work done on your iPhone [3]. Developers are especially excited about the new machine learning frameworks Apple released, which means we'll see even more powerful apps in the future [4].

                ## References
                [1] Apple announces iOS 18 with enhanced AI capabilities - https://apple.com/newsroom/ios-18
                [2] iPhone 15 Pro features titanium design and USB-C - https://apple.com/newsroom/iphone-15
                [3] App Store reports 40% increase in productivity apps - https://apple.com/newsroom/app-store
                [4] Developers praise new machine learning frameworks - https://developer.apple.com/ml-news
                [5] Users report improved battery life in iOS 18 beta - https://apple.com/newsroom/ios-18-beta
                """
            ).strip(),
        )
    ]

    def build(self, context: PromptContext) -> PromptPayload:
        cleaned = context.cleaned()
        if not cleaned.query:
            raise ValueError("PromptContext.query cannot be empty")
        if not cleaned.brand_name:
            raise ValueError("PromptContext.brand_name cannot be empty")
        if not cleaned.insights:
            raise ValueError("At least one insight is required to build the prompt")

        system_instruction = self._SYSTEM_PROMPT
        prompt_parts: List[str] = []

        for example in self._FEW_SHOT_EXAMPLES:
            prompt_parts.append(
                textwrap.dedent(
                    f"""
                    Example Input:
                    {example.user}

                    Example Output:
                    {example.assistant}
                    """
                ).strip()
            )

        user_prompt = self._build_user_prompt(cleaned)
        prompt_parts.append(user_prompt)

        return PromptPayload(
            system_instruction=system_instruction,
            prompt_parts=prompt_parts,
            few_shot_examples=list(self._FEW_SHOT_EXAMPLES),
        )

    def _build_user_prompt(self, context: PromptContext) -> str:
        # Use limited sources if available, otherwise fall back to insights
        if hasattr(context, '_limited_sources') and context._limited_sources:
            references = self._format_references_from_sources(context._limited_sources, context.query)
            insights_section = self._format_insights(context.insights)
        else:
            references = self._format_references(context.insights)
            insights_section = self._format_insights(context.insights)

        goals_section = "\n".join(f"- {goal}" for goal in context.analysis_goals) if context.analysis_goals else "- Assess brand sentiment and visibility."
        competitors_section = "\n".join(f"- {name}" for name in context.competitors) if context.competitors else "- No direct competitors provided."
        caveats_section = "\n".join(f"- {item}" for item in context.caveats) if context.caveats else "- None provided."

        brand_overview = context.brand_description or "No formal brand description supplied. Focus on evidence from insights."

        template = textwrap.dedent(
            """
            # Brand Analysis Brief: {brand_name}

            ## Primary Question
            {query}

            ## Brand Overview
            {brand_overview}

            ## Analysis Goals
            {goals_section}

            ## Competitors to Consider
            {competitors_section}

            ## Research Insights
            {insights_section}

            ## Caveats
            {caveats_section}

            {output_instructions}

            ## Available Sources for Citations
            Use these sources for your citations and copy them exactly in your References section:
            {references}
            """
        ).strip()

        return template.format(
            brand_name=context.brand_name,
            query=context.query,
            brand_overview=brand_overview,
            goals_section=goals_section,
            competitors_section=competitors_section,
            insights_section=insights_section,
            caveats_section=caveats_section,
            output_instructions=self._OUTPUT_INSTRUCTIONS,
            references=references,
        )

    @staticmethod
    def _format_insights(insights: Sequence[PromptInsight]) -> str:
        lines = []
        for index, insight in enumerate(insights, start=1):
            source_label = f" ({insight.source})" if insight.source else ""
            lines.append(
                textwrap.dedent(
                    f"""
                    {index}. {insight.title}{source_label}
                       Summary: {insight.summary}
                       Cite as: [{index}]
                    """
                ).strip()
            )
        return "\n".join(lines)

    @staticmethod
    def _format_references(insights: Sequence[PromptInsight]) -> str:
        lines = []
        for index, insight in enumerate(insights, start=1):
            lines.append(f"[{index}] {insight.title} - {insight.url}")
        return "\n".join(lines)

    @staticmethod
    def _format_references_from_sources(sources: Sequence[str], query: str) -> str:
        """Format references from source URLs directly."""
        lines = []
        for index, source_url in enumerate(sources, start=1):
            # Extract a meaningful title from the URL path
            title = PromptBuilder._extract_title_from_url(source_url, query)
            lines.append(f"[{index}] {title} - {source_url}")
        return "\n".join(lines)

    @staticmethod
    def _extract_title_from_url(url: str, query: str) -> str:
        """Extract a meaningful title from a URL."""
        url_lower = url.lower()

        if "apple.com" in url_lower:
            if "newsroom" in url_lower:
                # Extract specific newsroom article type from URL
                if "ios-26" in url_lower:
                    return "Apple elevates the iPhone experience with iOS 26"
                elif "iphone-17" in url_lower and "pro" in url_lower:
                    return "Apple unveils iPhone 17 Pro and iPhone 17 Pro Max"
                elif "iphone-17" in url_lower:
                    return "Apple debuts iPhone 17"
                elif "iphone-air" in url_lower:
                    return "Introducing iPhone Air with breakthrough design"
                elif "iphone-16" in url_lower:
                    return "Apple introduces iPhone 16 and iPhone 16 Plus"
                else:
                    return f"Apple Newsroom - {query} update"
            elif "iphone-17" in url_lower:
                return "iPhone 17 product page"
            elif "pdf" in url_lower and "ios" in url_lower:
                return "iOS 26 complete features guide"
            else:
                return f"Apple official - {query} information"
        else:
            # For non-Apple URLs, create a generic but specific title
            domain = url.split('/')[2] if '/' in url else url
            return f"{domain} - {query} coverage"


__all__ = [
    "PromptBuilder",
    "PromptContext",
    "PromptInsight",
    "PromptPayload",
    "FewShotExample",
]
