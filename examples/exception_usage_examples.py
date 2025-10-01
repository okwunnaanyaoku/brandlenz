#!/usr/bin/env python3
"""
BrandLens Exception Usage Examples

This module demonstrates how to use the custom exceptions in the BrandLens
brand visibility analyzer for robust error handling and debugging.

Run this script to see examples of:
- API integration error handling
- Data validation error handling
- Retry logic with error classification
- Error context and debugging information
"""

import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.exceptions import (
    BrandLensError,
    GeminiAPIError,
    TavilyAPIError,
    RateLimitError,
    ValidationError,
    BrandAnalysisError,
    InvalidBrandError,
    classify_error_for_retry,
    get_user_friendly_message,
    create_error_context,
)


def simulate_gemini_api_call(api_key: str, prompt: str) -> Dict[str, Any]:
    """Simulate a Gemini API call with potential errors."""

    # Simulate authentication error
    if not api_key or api_key == "invalid":
        raise GeminiAPIError(
            "Invalid API key provided",
            model="gemini-1.5-flash",
            status_code=401,
            error_code="GEMINI_AUTH_ERROR",
            user_message="Please check your Gemini API key configuration.",
            retryable=False,
        )

    # Simulate rate limiting
    if "rate_limit" in prompt.lower():
        raise RateLimitError(
            "Rate limit exceeded for Gemini API",
            service="gemini",
            retry_after=60,
            quota_type="requests",
            current_usage=1000,
            quota_limit=1000,
            user_message="API rate limit exceeded. Please try again in 60 seconds.",
        )

    # Simulate successful response
    return {
        "content": f"Analysis for: {prompt}",
        "tokens": 150,
        "model": "gemini-1.5-flash"
    }


def simulate_tavily_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Simulate a Tavily search with potential errors."""

    # Simulate invalid query
    if not query.strip():
        raise TavilyAPIError(
            "Empty search query provided",
            query=query,
            search_depth="advanced",
            max_results=max_results,
            error_code="TAVILY_EMPTY_QUERY",
            user_message="Please provide a valid search query.",
            retryable=False,
        )

    # Simulate timeout
    if "timeout" in query.lower():
        raise TavilyAPIError(
            "Search request timed out",
            query=query,
            status_code=408,
            user_message="Search took too long. Please try a simpler query.",
            retryable=True,
        )

    # Simulate successful response
    return {
        "results": [
            {"url": f"https://example.com/{i}", "title": f"Result {i}"}
            for i in range(max_results)
        ]
    }


def validate_brand_data(brand_name: str, brand_domain: str) -> None:
    """Validate brand data with custom exceptions."""

    if not brand_name or len(brand_name.strip()) < 2:
        raise InvalidBrandError(
            "Brand name must be at least 2 characters long",
            brand_name=brand_name,
            validation_issue="insufficient_length",
            user_message="Please provide a valid brand name.",
        )

    if not brand_domain or "." not in brand_domain:
        raise InvalidBrandError(
            "Brand domain must be a valid domain format",
            brand_name=brand_name,
            brand_domain=brand_domain,
            validation_issue="invalid_domain_format",
            user_message="Please provide a valid domain (e.g., example.com).",
        )


def analyze_brand_with_retry(
    brand_name: str,
    brand_domain: str,
    query: str,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Perform brand analysis with comprehensive error handling and retry logic.

    This demonstrates how to use BrandLens exceptions in a real analysis pipeline.
    """

    for attempt in range(max_retries + 1):
        try:
            print(f"\n--- Attempt {attempt + 1} ---")

            # Step 1: Validate brand data
            print("🔍 Validating brand data...")
            validate_brand_data(brand_name, brand_domain)
            print("✅ Brand data validated")

            # Step 2: Perform search
            print("🔍 Performing Tavily search...")
            search_results = simulate_tavily_search(query)
            print(f"✅ Found {len(search_results['results'])} search results")

            # Step 3: Generate analysis with Gemini
            print("🔍 Generating analysis with Gemini...")
            analysis = simulate_gemini_api_call("valid_key", query)
            print("✅ Analysis generated successfully")

            # Return successful result
            return {
                "brand_name": brand_name,
                "brand_domain": brand_domain,
                "search_results": search_results,
                "analysis": analysis,
                "attempts": attempt + 1,
            }

        except BrandLensError as e:
            print(f"❌ BrandLens Error: {e}")
            print(f"   Error Code: {e.error_code}")
            print(f"   User Message: {e.user_message}")
            print(f"   Retryable: {e.retryable}")
            print(f"   Context: {e.context}")

            # Check if we should retry
            if not classify_error_for_retry(e) or attempt >= max_retries:
                print(f"🛑 Giving up after {attempt + 1} attempts")

                # Create final error with analysis context
                raise BrandAnalysisError(
                    f"Brand analysis failed after {attempt + 1} attempts: {e.message}",
                    analysis_stage="pipeline_execution",
                    brand_name=brand_name,
                    query=query,
                    cause=e,
                    context=create_error_context(
                        "brand_analysis_pipeline",
                        attempts=attempt + 1,
                        max_retries=max_retries,
                        original_error=e.error_code,
                    ),
                )
            else:
                # Wait before retry (exponential backoff)
                wait_time = 2 ** attempt
                print(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        except Exception as e:
            print(f"❌ Unexpected Error: {e}")

            # Wrap unexpected errors in our exception system
            wrapped_error = BrandAnalysisError(
                f"Unexpected error during brand analysis: {e}",
                analysis_stage="unknown",
                brand_name=brand_name,
                query=query,
                cause=e,
                context=create_error_context(
                    "brand_analysis_pipeline",
                    unexpected_error=True,
                    original_error_type=type(e).__name__,
                ),
            )

            if attempt >= max_retries:
                raise wrapped_error
            else:
                print(f"⏳ Retrying unexpected error...")
                time.sleep(1)


def demonstrate_error_handling():
    """Demonstrate comprehensive error handling scenarios."""

    print("=" * 60)
    print("BrandLens Exception Handling Demonstration")
    print("=" * 60)

    # Test cases with different error scenarios
    test_cases = [
        {
            "name": "Successful Analysis",
            "brand_name": "TechCorp",
            "brand_domain": "techcorp.com",
            "query": "TechCorp software solutions",
        },
        {
            "name": "Invalid Brand Name",
            "brand_name": "A",  # Too short
            "brand_domain": "invalid.com",
            "query": "brand analysis",
        },
        {
            "name": "Rate Limit Error",
            "brand_name": "TestBrand",
            "brand_domain": "testbrand.com",
            "query": "rate_limit test query",  # Triggers rate limit
        },
        {
            "name": "Timeout Error",
            "brand_name": "TimeoutBrand",
            "brand_domain": "timeout.com",
            "query": "timeout test query",  # Triggers timeout
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*20} Test Case {i}: {test_case['name']} {'='*20}")

        try:
            result = analyze_brand_with_retry(
                brand_name=test_case["brand_name"],
                brand_domain=test_case["brand_domain"],
                query=test_case["query"],
                max_retries=2,
            )

            print(f"🎉 SUCCESS: Analysis completed in {result['attempts']} attempts")
            print(f"   Found {len(result['search_results']['results'])} results")

        except BrandLensError as e:
            print(f"💥 FINAL FAILURE: {e.error_code}")
            print(f"   User Message: {get_user_friendly_message(e)}")
            print(f"   Technical Details: {e.message}")

            # Demonstrate error serialization for logging
            error_data = e.to_dict()
            print(f"   Logged Error Context: {len(error_data)} fields captured")

        except Exception as e:
            print(f"💥 UNEXPECTED ERROR: {e}")

    print("\n" + "=" * 60)
    print("Demonstration completed!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_error_handling()