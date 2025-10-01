# BrandLens Makefile
# Development and deployment automation

.PHONY: install test test-unit test-integration test-e2e run clean lint format validate-apis benchmark help

# =============================================================================
# INSTALLATION AND SETUP
# =============================================================================

install:
	@echo "Installing BrandLens dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev]"
	python -m spacy download en_core_web_sm
	@echo "✅ Installation complete!"

install-prod:
	@echo "Installing production dependencies only..."
	pip install --upgrade pip
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	@echo "✅ Production installation complete!"

# =============================================================================
# TESTING
# =============================================================================

test:
	@echo "Running all tests..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "✅ All tests completed!"

test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v
	@echo "✅ Unit tests completed!"

test-integration:
	@echo "Running integration tests with REAL APIs..."
	pytest tests/integration/ -v -m api --tb=short
	@echo "✅ Integration tests completed!"

test-e2e:
	@echo "Running end-to-end tests with REAL APIs..."
	pytest tests/e2e/ -v -m "api and slow" --tb=short
	@echo "✅ End-to-end tests completed!"

test-fast:
	@echo "Running fast tests only (no API calls)..."
	pytest tests/unit/ -v -x
	@echo "✅ Fast tests completed!"

# =============================================================================
# CODE QUALITY
# =============================================================================

lint:
	@echo "Running linting checks..."
	flake8 src tests --max-line-length=88 --extend-ignore=E203,W503
	mypy src --strict
	@echo "✅ Linting completed!"

format:
	@echo "Formatting code..."
	black src tests
	@echo "✅ Code formatting completed!"

format-check:
	@echo "Checking code formatting..."
	black --check src tests
	@echo "✅ Format check completed!"

# =============================================================================
# API VALIDATION
# =============================================================================

validate-apis:
	@echo "Validating API connections..."
	python scripts/validate_api_keys.py
	@echo "✅ API validation completed!"

# =============================================================================
# RUNNING THE APPLICATION
# =============================================================================

run:
	@echo "Running BrandLens analysis..."
	python -m src --brand "$(BRAND)" --url "$(URL)" --question "$(QUESTION)"

run-example:
	@echo "Running example analysis..."
	python -m src \
		--brand "Apple" \
		--url "apple.com" \
		--question "What are the latest iPhone features?" \
		--max-searches 3 \
		--max-sources 5 \
		--output examples/output_samples/apple_example.json

run-debug:
	@echo "Running with debug output..."
	DEBUG=true python -m src --brand "$(BRAND)" --url "$(URL)" --question "$(QUESTION)" --verbose

# =============================================================================
# PERFORMANCE AND MONITORING
# =============================================================================

benchmark:
	@echo "Running performance benchmarks..."
	python scripts/benchmark.py
	@echo "✅ Benchmarking completed!"

cost-analysis:
	@echo "Analyzing API costs..."
	python scripts/cost_calculator.py
	@echo "✅ Cost analysis completed!"

monitor:
	@echo "Starting performance monitoring..."
	python -m src monitor --duration 3600  # Monitor for 1 hour

# =============================================================================
# DEVELOPMENT UTILITIES
# =============================================================================

clean:
	@echo "Cleaning up project..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	rm -rf build dist *.egg-info
	rm -rf .cache .test_cache
	@echo "✅ Cleanup completed!"

reset-cache:
	@echo "Resetting cache..."
	rm -rf .cache .test_cache
	@echo "✅ Cache reset completed!"

# =============================================================================
# DOCKER SUPPORT (Future)
# =============================================================================

docker-build:
	@echo "Building Docker image..."
	docker build -t brandlens:latest .

docker-run:
	@echo "Running in Docker..."
	docker run --env-file .env brandlens:latest \
		--brand "$(BRAND)" --url "$(URL)" --question "$(QUESTION)"

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs:
	@echo "Generating documentation..."
	# Future: Add documentation generation
	@echo "📚 Documentation generation not implemented yet"

# =============================================================================
# CONTINUOUS INTEGRATION
# =============================================================================

ci-test:
	@echo "Running CI test suite..."
	make format-check
	make lint
	make test-unit
	@echo "✅ CI tests completed!"

ci-integration:
	@echo "Running CI integration tests..."
	make validate-apis
	make test-integration
	@echo "✅ CI integration tests completed!"

# =============================================================================
# HELP
# =============================================================================

help:
	@echo "BrandLens Makefile Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install          Install all dependencies and setup"
	@echo "  install-prod     Install production dependencies only"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests with coverage"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests (REAL APIs)"
	@echo "  test-e2e         Run end-to-end tests (REAL APIs)"
	@echo "  test-fast        Run fast tests without API calls"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run linting and type checking"
	@echo "  format           Format code with black"
	@echo "  format-check     Check code formatting"
	@echo ""
	@echo "Running:"
	@echo "  run              Run analysis (set BRAND, URL, QUESTION)"
	@echo "  run-example      Run example Apple analysis"
	@echo "  run-debug        Run with debug output"
	@echo ""
	@echo "Validation:"
	@echo "  validate-apis    Test API connections"
	@echo ""
	@echo "Performance:"
	@echo "  benchmark        Run performance benchmarks"
	@echo "  cost-analysis    Analyze API costs"
	@echo "  monitor          Start performance monitoring"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            Clean up temporary files"
	@echo "  reset-cache      Reset application cache"
	@echo "  help             Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make run BRAND='Tesla' URL='tesla.com' QUESTION='How safe is autopilot?'"
	@echo "  make test-integration"
	@echo "  make benchmark"