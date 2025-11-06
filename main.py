import argparse
import logging
from code.preprocessing.datamodule import DataModule
from code.preprocessing.logging_configuration import setup_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process visual reasoning datasets")
    parser.add_argument(
        "--config", 
        type=str, 
        default="code/preprocessing/dataset_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--download", 
        action="store_true",
        help="Download datasets from HuggingFace (only if hf_repo_id is specified in config)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging with specified level
    logger = setup_logging(getattr(logging, args.log_level))
    
    # Create and run data module
    data_module = DataModule(
        config_path=args.config,
        load_from_hf=args.download
    )
    
    data_module.run()
