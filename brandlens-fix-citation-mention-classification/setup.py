"""
BrandLens Setup Configuration
Advanced Brand Visibility Analysis for LLM Responses
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="brandlens",
    version="1.1.0",
    author="BrandLens Team",
    author_email="contact@brandlens.dev",
    description="Advanced Brand Visibility Analysis for LLM Responses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/brandlens",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        # Core dependencies
        "click>=8.1.0",
        "rich>=13.7.0",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",

        # LLM and Search APIs
        "google-generativeai>=0.3.0",
        "tavily-python>=0.3.0",

        # Token optimization
        "tiktoken>=0.5.0",

        # NLP and extraction
        "spacy>=3.7.0",

        # Async and HTTP
        "httpx>=0.25.0",
        "aiofiles>=23.2.0",

        # Utilities
        "tenacity>=8.2.0",
        "jsonschema>=4.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
            "pre-commit>=3.5.0",
        ],
        "test": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.12.0",
            "responses>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "brandlens=src.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["py.typed"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/brandlens/issues",
        "Source": "https://github.com/yourusername/brandlens",
        "Documentation": "https://brandlens.readthedocs.io/",
    },
)