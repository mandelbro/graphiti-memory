# Just command runner configuration
# Docs: https://github.com/casey/just

# Load variables from .env automatically
set dotenv-load := true

# Use bash for recipes
set shell := ["bash", "-cu"]

# Tool variables
UV := "uv"
DC := "docker compose"

# Default task
default: check

# Install production dependencies
install:
	{{UV}} sync

# Install development dependencies (ruff, pyright, pytest, black, etc.)
install-dev:
	{{UV}} sync --extra dev

# Format imports and code (ruff)
fmt:
	{{UV}} run ruff check --select I --fix
	{{UV}} run ruff format

# Alias for format
format: fmt

# Lint (ruff) and static type check (pyright)
lint:
	{{UV}} run ruff check
	{{UV}} run pyright .

# Typecheck only
typecheck:
	{{UV}} run pyright .

# Run tests (pass additional args after --)
# Usage: just test -- -k "pattern" -q
@test *ARGS:
	{{UV}} run pytest {{ARGS}}

# Run a specific test file
# Usage: just test-file tests/test_basic.py
@test-file FILE:
	{{UV}} run pytest {{FILE}} -v

# Run tests with coverage
@test-cov:
	{{UV}} run pytest --cov=src --cov-report=term-missing --cov-report=html

# Quick unit subset
@test-unit:
	{{UV}} run pytest tests/test_basic.py tests/test_oauth_simple.py

# OAuth-focused tests with coverage on oauth_wrapper
@test-oauth:
	{{UV}} run pytest tests/test_oauth_simple.py --cov=src/oauth_wrapper --cov-report=term-missing

# Clean build/test artifacts
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .coverage coverage.xml

# Compose helpers
@dev:
	{{DC}} -f docker-compose.yml -f docker-compose.dev.yml up --build

@up:
	{{DC}} up -d

@down:
	{{DC}} down

@rebuild:
	{{DC}} down && {{DC}} build --no-cache && {{DC}} up -d && {{DC}} logs -f

@restart:
	{{DC}} down && {{DC}} up -d && {{DC}} logs -f

@logs:
	{{DC}} logs -f

# Run everything commonly needed locally
check: fmt lint typecheck test
