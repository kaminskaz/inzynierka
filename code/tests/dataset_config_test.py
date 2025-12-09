from code.technical.utils import get_dataset_config
import pprint
from dataclasses import asdict, is_dataclass

def pretty_print_config(config):
    """
    Pretty-print all attributes of a ModelConfig (dataclass or normal class).
    """

    if config is None:
        print("Config is None")
        return

    if isinstance(config, dict):
        pprint.pprint(config)
        return

    try:
        if is_dataclass(config):
            pprint.pprint(asdict(config))
            return
    except Exception:
        pass



def get_dataset_config_test():
    datasets = [
        "raven",
        "cvr",
        "xyz"
    ]

    print("\n=== Running pairwise get_model_config tests ===\n")

    for i, dataset in enumerate(datasets):
        print(f"\n--- Test {i+1}: dataset='{dataset}' ---")
        try:
            config = get_dataset_config(dataset)
            pretty_print_config(config)
        except Exception as e:
            print(f"ERROR: {e}")
    print("\n=== Test complete ===")

if __name__ == "__main__":
    get_dataset_config_test()





    