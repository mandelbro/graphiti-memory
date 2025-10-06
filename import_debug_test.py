#!/usr/bin/env python3
"""
Ultra-minimal debug test to isolate import hangs.
"""

import signal
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def timeout_handler(signum, frame):
    print(f"TIMEOUT: Import took too long!")
    sys.exit(1)

def test_import(module_name, import_statement):
    """Test a single import with timeout."""
    print(f"Testing import: {module_name}")
    print(f"  Statement: {import_statement}")

    # Set 5-second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)

    try:
        exec(import_statement)
        signal.alarm(0)  # Cancel timeout
        print(f"  ✓ SUCCESS: {module_name}")
        return True
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        print(f"  ✗ ERROR: {module_name} - {e}")
        return False

def main():
    """Test imports one by one."""
    print("=" * 60)
    print("IMPORT DEBUG TEST")
    print("=" * 60)

    # Test basic imports first
    imports_to_test = [
        ("Standard Library", "import asyncio, logging, os"),
        ("Pydantic", "from pydantic import BaseModel"),
        ("FastMCP", "from mcp.server.fastmcp import FastMCP"),
        ("GraphitiCore", "from graphiti_core import Graphiti"),

        # Our modules
        ("src.__init__", "import src"),
        ("src.config", "from src.config import GraphitiConfig"),
        ("src.config.llm_config", "from src.config.llm_config import GraphitiLLMConfig"),
        ("src.utils", "from src.utils import initialization_state"),
        ("src.initialization", "from src.initialization import graphiti_client"),
        ("src.tools", "from src.tools import memory_tools"),
        ("src.utils.queue_utils", "from src.utils.queue_utils import episode_queues"),
    ]

    results = {}

    for module_name, import_statement in imports_to_test:
        result = test_import(module_name, import_statement)
        results[module_name] = result

        if not result:
            print(f"\n!!! STOPPING: Failed at {module_name} !!!")
            break

    print("\n" + "=" * 60)
    print("IMPORT TEST RESULTS:")
    for module_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {module_name}: {status}")

    print("=" * 60)
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        print(f"\nImport debug test finished with exit code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nTest failed with exception: {e}")
        sys.exit(1)
