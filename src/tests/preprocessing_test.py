import argparse
import logging
from src.preprocessing.data_module import DataModule
from src.technical.utils import setup_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process visual reasoning datasets")
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/technical/configs/dataset_config.json",
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
    
    logger = setup_logging(getattr(logging, args.log_level))
    
    data_module = DataModule(
        config_path=args.config,
        load_from_hf=args.download
    )
    
    data_module.run()
