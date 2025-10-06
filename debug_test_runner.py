#!/usr/bin/env python3
"""
Debug test runner for the Graphiti pipeline integration tests.

This script runs specific tests with detailed logging to identify where
the execution is hanging or getting stuck.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_test.log', mode='w')
    ]
)

# Set specific loggers to DEBUG
logger = logging.getLogger(__name__)
logging.getLogger("src").setLevel(logging.DEBUG)
logging.getLogger("tests").setLevel(logging.DEBUG)
logging.getLogger("graphiti_core").setLevel(logging.DEBUG)


def run_smoke_test():
    """Run the smoke test with detailed logging."""
    logger.info("Starting debug test run")

    # Import pytest and run the specific test
    import pytest

    # Set environment variables for testing
    os.environ.update({
        "TEST_NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "TEST_NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
        "TEST_NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "demodemo"),
        "TEST_OLLAMA_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    })

    logger.info("Environment variables set for testing")
    logger.info(f"Neo4j URI: {os.environ.get('TEST_NEO4J_URI')}")
    logger.info(f"Ollama URL: {os.environ.get('TEST_OLLAMA_URL')}")

    # Run the smoke test specifically
    test_file = "tests/test_graphiti_pipeline_integration.py"
    test_method = "TestGraphitiPipelineIntegration::test_smoke_basic_initialization"

    logger.info(f"Running test: {test_file}::{test_method}")

    # Run pytest with verbose output
    exit_code = pytest.main([
        test_file + "::" + test_method,
        "-v",
        "-s",  # Don't capture output
        "--tb=long",  # Long traceback format
        "--log-cli-level=DEBUG",  # Show debug logs in CLI
        "--no-header",  # Skip pytest header
        "-x",  # Stop on first failure
    ])

    logger.info(f"Test completed with exit code: {exit_code}")
    return exit_code


def run_lifecycle_test():
    """Run the lifecycle test with detailed logging."""
    logger.info("Starting lifecycle test run")

    import pytest

    # Run the lifecycle test specifically
    test_file = "tests/test_graphiti_pipeline_integration.py"
    test_method = "TestGraphitiPipelineIntegration::test_memory_operation_complete_lifecycle"

    logger.info(f"Running test: {test_file}::{test_method}")

    # Run pytest with verbose output and timeout
    exit_code = pytest.main([
        test_file + "::" + test_method,
        "-v",
        "-s",  # Don't capture output
        "--tb=long",  # Long traceback format
        "--log-cli-level=DEBUG",  # Show debug logs in CLI
        "--no-header",  # Skip pytest header
        "-x",  # Stop on first failure
        "--timeout=60",  # 60 second timeout
    ])

    logger.info(f"Test completed with exit code: {exit_code}")
    return exit_code


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Debug test runner for Graphiti pipeline")
    parser.add_argument("--test", choices=["smoke", "lifecycle", "both"], default="smoke",
                       help="Which test to run")
    parser.add_argument("--timeout", type=int, default=60,
                       help="Test timeout in seconds")

    args = parser.parse_args()

    logger.info(f"Debug test runner starting - test={args.test}, timeout={args.timeout}")

    try:
        if args.test == "smoke":
            return run_smoke_test()
        elif args.test == "lifecycle":
            return run_lifecycle_test()
        elif args.test == "both":
            smoke_result = run_smoke_test()
            if smoke_result == 0:
                logger.info("Smoke test passed, running lifecycle test")
                return run_lifecycle_test()
            else:
                logger.error("Smoke test failed, skipping lifecycle test")
                return smoke_result
    except KeyboardInterrupt:
        logger.warning("Test run interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Test run failed with exception: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    print(f"\nDebug test runner finished with exit code: {exit_code}")
    sys.exit(exit_code)
