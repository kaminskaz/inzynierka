from code.technical.utils import get_model_config, get_dataset_config
import pprint

def pretty_print_config(config):
    """
    Pretty-print all attributes of a ModelConfig (dataclass or normal class).
    """
    try:
        from dataclasses import asdict, is_dataclass
        if is_dataclass(config):
            pprint.pprint(asdict(config))
            return
    except Exception:
        pass
    attrs = {
        k: v for k, v in vars(config).items()
        if not k.startswith("_")
    }
    pprint.pprint(attrs)


def get_model_config_test():
    models = [
        "OpenGVLab/InternVL3-8B",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "MissingModel"
    ]

    param_set_options = ["1", 1, "something different", None, 4]

    print("\n=== Running pairwise get_model_config tests ===\n")

    for i, (model, param) in enumerate(zip(models, param_set_options)):
        print(f"\n--- Test {i+1}: model='{model}' param_set='{param}' ---")
        try:
            if i != 3:
                config = get_model_config(model, param)
            else:
                config = get_model_config(model)
            pretty_print_config(config)
        except Exception as e:
            print(f"ERROR: {e}")
    print("\n=== Test complete ===")

def get_dataset_config_test():
    datasets = [
        "raven",
        "cvr",
        "xyz"
    ]

    print("\n=== Running pairwise get_dataset_config tests ===\n")

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
    get_model_config_test()





    