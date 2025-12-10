import logging
from typing import Any

# Import your factories
# Ensure this script is in the root directory so these imports work
from code.ensemble.ensemble_factory import EnsembleFactory
from code.preprocessing.processor_factory import ProcessorFactory
from code.strategies.strategy_factory import StrategyFactory

# Setup basic logging to see the factory outputs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestRunner")

# --------------------------------------------------------------------------
# Helper Classes
# --------------------------------------------------------------------------
class DummyConfig:
    """A minimal config object to satisfy the ProcessorFactory check."""
    def __init__(self, category: str):
        self.category = category

class DummyModel:
    """A minimal model object to satisfy StrategyFactory logging."""
    def get_model_name(self):
        return "DummyModel"

# --------------------------------------------------------------------------
# Test Functions
# --------------------------------------------------------------------------

def test_processor_missing_sheetmaker():
    print("\n--- Testing ProcessorFactory: Missing Sheetmaker ---")
    
    # 1. Create a config that triggers the "standard" logic
    # The factory checks: if config.category in ["standard", "choice_only"]
    config = DummyConfig(category="standard")

    try:
        # 2. Call create_processor without a sheet_maker (None)
        ProcessorFactory.create_processor(config, sheet_maker=None)
        print("FAILED: ProcessorFactory did not raise ValueError.")
    except ValueError as e:
        # 3. Verify the error message matches the code
        print(f"SUCCESS: Caught expected error: {e}")
    except Exception as e:
        print(f"FAILED: Caught unexpected exception type: {type(e)}")


def test_ensemble_unknown_type():
    print("\n--- Testing EnsembleFactory: Unknown Type ---")
    
    factory = EnsembleFactory()
    
    # 1. Define arguments
    dataset_name = "test_dataset"
    members_config = [["majority", "gpt-4", "v1"]]
    bad_type = "galactic_voting" # This type is not in the ensemble_map

    try:
        # 2. Call create_ensemble with the unknown type
        factory.create_ensemble(
            dataset_name=dataset_name, 
            members_configuration=members_config, 
            type_name=bad_type
        )
        print("FAILED: EnsembleFactory did not raise ValueError.")
    except ValueError as e:
        # 3. Verify the error message
        print(f"SUCCESS: Caught expected error: {e}")


def test_strategy_unknown_strategy():
    print("\n--- Testing StrategyFactory: Unknown Strategy ---")
    
    factory = StrategyFactory()
    model = DummyModel()
    
    # 1. Define invalid strategy name
    bad_strategy = "mind_reading" # Not in strategy_map

    try:
        # 2. Call create_strategy
        factory.create_strategy(
            dataset_name="test_dataset",
            strategy_name=bad_strategy,
            model_object=model,
            results_dir="./results"
        )
        print("FAILED: StrategyFactory did not raise ValueError.")
    except ValueError as e:
        # 3. Verify the error message
        print(f"SUCCESS: Caught expected error: {e}")


def test_strategy_no_dataset_config():
    print("\n--- Testing StrategyFactory: No Dataset Config ---")
    
    factory = StrategyFactory()
    model = DummyModel()
    
    # 1. Use a valid strategy name so we pass the first check
    valid_strategy = "classification" 
    
    # 2. Use a dataset name that definitely does not exist in your config JSON
    bad_dataset = "non_existent_dataset_12345"

    try:
        # 3. Call create_strategy
        # This should fail when calling get_dataset_config internally
        factory.create_strategy(
            dataset_name=bad_dataset,
            strategy_name=valid_strategy,
            model_object=model,
            results_dir="./results"
        )
        print("FAILED: StrategyFactory did not raise ValueError.")
    except ValueError as e:
        # 4. Verify the error message
        print(f"SUCCESS: Caught expected error: {e}")
    except Exception as e:
        # Note: If get_dataset_config raises a different error (like FileNotFoundError), catch it here
        print(f"NOTE: Caught {type(e)}: {e}")


# --------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------
if __name__ == "__main__":
    test_processor_missing_sheetmaker()
    test_ensemble_unknown_type()
    test_strategy_unknown_strategy()
    test_strategy_no_dataset_config()