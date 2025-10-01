import argparse
from code.preprocessing.dataprocessor import DataProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data preprocessing.")
    parser.add_argument(
        "--preprocess_data",
        action="store_true",
        help="Run preprocessing if this flag is provided."
    )
    args = parser.parse_args()

    if args.preprocess_data:
        print("Starting data preprocessing...")
        processor = DataProcessor(load=False)
        processor.process()
        print("Data preprocessing completed.")
    else:
        print("Preprocessing skipped.")
