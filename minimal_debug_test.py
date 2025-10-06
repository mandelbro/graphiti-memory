#!/usr/bin/env python3
"""
Minimal debug test to isolate hanging issues without external dependencies.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_imports():
    """Test that we can import our modules without hanging."""
    logger.info("=== Testing basic imports ===")

    try:
        logger.info("Importing src.config...")
        from src.config import GraphitiConfig, GraphitiLLMConfig
        logger.info("✓ Config imports successful")

        logger.info("Importing src.initialization.graphiti_client...")
        from src.initialization.graphiti_client import create_graphiti_client
        logger.info("✓ Graphiti client imports successful")

        logger.info("Importing src.utils.initialization_state...")
        from src.utils.initialization_state import initialization_manager
        logger.info("✓ Initialization state imports successful")

        logger.info("Importing src.tools.memory_tools...")
        from src.tools.memory_tools import add_memory
        logger.info("✓ Memory tools imports successful")

        return True

    except Exception as e:
        logger.error(f"Import failed: {e}")
        return False


async def test_config_creation():
    """Test that we can create configurations without hanging."""
    logger.info("=== Testing config creation ===")

    try:
        from src.config import GraphitiConfig, GraphitiLLMConfig, Neo4jConfig

        logger.info("Creating test config...")
        config = GraphitiConfig(
            neo4j=Neo4jConfig(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test",
            ),
            llm=GraphitiLLMConfig(
                model="test-model",
                use_ollama=True,
                ollama_base_url="http://localhost:11434/v1",
                ollama_llm_model="test-model",
                api_key="test",
            ),
            use_custom_entities=True,
            group_id="test",
        )
        logger.info("✓ Config creation successful")
        logger.info(f"Config neo4j uri: {config.neo4j.uri}")
        return True

    except Exception as e:
        logger.error(f"Config creation failed: {e}")
        return False


async def test_initialization_manager():
    """Test initialization manager without actually initializing services."""
    logger.info("=== Testing initialization manager ===")

    try:
        from src.utils.initialization_state import initialization_manager

        logger.info("Resetting initialization manager...")
        await initialization_manager.reset_for_testing()
        logger.info("✓ Reset successful")

        logger.info("Starting initialization...")
        await initialization_manager.start_initialization()
        logger.info("✓ Start successful")

        logger.info("Checking state...")
        state = initialization_manager.get_state()
        logger.info(f"State: {state}")

        logger.info("Failing initialization (test cleanup)...")
        await initialization_manager.fail_initialization("test cleanup")
        logger.info("✓ Fail successful")

        return True

    except Exception as e:
        logger.error(f"Initialization manager test failed: {e}")
        return False


async def test_queue_utils():
    """Test queue utilities without starting actual workers."""
    logger.info("=== Testing queue utilities ===")

    try:
        from src.utils.queue_utils import episode_queues, queue_workers

        logger.info("Clearing queues...")
        episode_queues.clear()
        queue_workers.clear()
        logger.info(f"✓ Queues cleared - episode_queues: {len(episode_queues)}, queue_workers: {len(queue_workers)}")

        logger.info("Testing queue creation...")
        test_group = "test-group"
        episode_queues[test_group] = asyncio.Queue()
        logger.info(f"✓ Queue created for {test_group}")

        logger.info("Testing queue cleanup...")
        del episode_queues[test_group]
        logger.info("✓ Queue cleanup successful")

        return True

    except Exception as e:
        logger.error(f"Queue utils test failed: {e}")
        return False


async def main():
    """Run all minimal tests."""
    logger.info("Starting minimal debug tests...")
    logger.info("=" * 60)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("Config Creation", test_config_creation),
        ("Initialization Manager", test_initialization_manager),
        ("Queue Utils", test_queue_utils),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n>>> Running {test_name} <<<")
        try:
            # Use asyncio.wait_for to add timeout to each test
            result = await asyncio.wait_for(test_func(), timeout=10.0)
            results[test_name] = result
            logger.info(f">>> {test_name}: {'PASS' if result else 'FAIL'} <<<")
        except asyncio.TimeoutError:
            logger.error(f">>> {test_name}: TIMEOUT <<<")
            results[test_name] = False
        except Exception as e:
            logger.error(f">>> {test_name}: ERROR - {e} <<<")
            results[test_name] = False

    logger.info("\n" + "=" * 60)
    logger.info("MINIMAL TEST RESULTS:")
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  {test_name}: {status}")

    all_passed = all(results.values())
    logger.info(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    logger.info("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        print(f"\nMinimal debug tests finished with exit code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nTests failed with exception: {e}")
        sys.exit(1)
