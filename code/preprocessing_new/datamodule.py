import os
import json
from pathlib import Path 
import traceback
from typing import Dict
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, login
from code.preprocessing_new.logging_configuration import setup_logging
from code.preprocessing_new.processorconfig import ProcessorConfig
from code.preprocessing_new.processorfactory import ProcessorFactory
import logging
import shutil

class DataModule:
    """Main data processing module."""
    
    def __init__(self, config_path: str = "code/preprocessing_new/dataset_config.json", load_from_hf: bool = False):
        self.config_path = Path(config_path)
        self.load_from_hf = load_from_hf
        self.logger = logging.getLogger("DataModule")
        self.configs = self.load_configs()
        self.processors = {}
        
        load_dotenv()
    
    def load_configs(self) -> Dict[str, ProcessorConfig]:
        """Load all dataset configurations."""
        try:
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
            
            configs = {}
            for name, cfg in config_dict.items():
                try:
                    configs[name] = ProcessorConfig.from_dict(cfg)
                except Exception as e:
                    self.logger.error(f"Error loading config for {name}: {e}")
            
            self.logger.info(f"Loaded {len(configs)} dataset configurations")
            return configs
            
        except Exception as e:
            self.logger.error(f"Error loading config file {self.config_path}: {e}")
            raise
    
    def setup(self) -> None:
        """Set up all processors."""
        # Import sheet maker (avoiding circular imports)
        from code.preprocessing_new.standardsheetmaker import StandardSheetMaker
        sheet_maker = StandardSheetMaker()
        
        for dataset_name, config in self.configs.items():
            try:
                self.processors[dataset_name] = ProcessorFactory.create_processor(
                    dataset_name, 
                    config, 
                    sheet_maker if config.category != "BP" else None
                )
                self.logger.debug(f"Created processor for {dataset_name}")
            except Exception as e:
                self.logger.error(f"Error creating processor for {dataset_name}: {e}")
        
        self.logger.info(f"Set up {len(self.processors)} processors")
    
    def download_datasets(self) -> None:
        """Download datasets from HuggingFace."""
        # Only download if datasets have hf_repo_id specified
        datasets_to_download = {
            name: config for name, config in self.configs.items()
            if config.hf_repo_id
        }
        
        if not datasets_to_download:
            self.logger.info("No datasets configured for HuggingFace download")
            return
        
        self.logger.info(f"Found {len(datasets_to_download)} datasets to download from HuggingFace")
        
        # Login to HuggingFace
        hf_token = os.getenv("HF_API_TOKEN")
        if not hf_token:
            self.logger.error("HF_API_TOKEN not found in environment variables")
            raise ValueError("HF_API_TOKEN required for downloading from HuggingFace")
        
        try:
            login(token=hf_token)
            self.logger.info("Successfully logged in to HuggingFace")
        except Exception as e:
            self.logger.error(f"Failed to login to HuggingFace: {e}")
            raise
        
        # Download each unique repo
        downloaded = set()
        for name, config in datasets_to_download.items():
            if config.hf_repo_id not in downloaded:
                try:
                    self.download_from_hf(config.hf_repo_id, config.hf_repo_type)
                    downloaded.add(config.hf_repo_id)
                except Exception as e:
                    self.logger.error(f"Failed to download {config.hf_repo_id}: {e}")
        
        self.logger.info(f"Downloaded {len(downloaded)} datasets from HuggingFace")
    
    def download_from_hf(self, repo_id: str, repo_type: str = "dataset") -> None:
        """Download a single dataset from HuggingFace."""
        data_path = Path("data_raw") / repo_id.split("/")[-1]
        
        # Check if already exists
        if data_path.exists() and any(data_path.iterdir()):
            self.logger.info(f"Dataset {repo_id} already exists at {data_path}, skipping download")
            return
        
        data_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Downloading {repo_id} to {data_path}...")
        
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=str(data_path),
                max_workers=3
            )
            self.logger.info(f"Successfully downloaded {repo_id}")
        except Exception as e:
            self.logger.error(f"Error downloading {repo_id}: {e}", exc_info=True)
            raise
    
    def process_all(self) -> None:
        """Process all datasets."""
        total_start_time = logging.time if hasattr(logging, 'time') else None
        
        for dataset_name, processor in self.processors.items():
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Processing dataset: {dataset_name}")
            self.logger.info(f"{'='*60}")
            
            try:
                processor.process()
            except Exception as e:
                self.logger.error(f"Failed to process {dataset_name}: {e}", exc_info=True)
        
        self.logger.info(f"{'='*60}")
        self.logger.info("All datasets processing complete")
        self.logger.info(f"{'='*60}")
    
    def run(self) -> None:
        """Run the complete data processing pipeline."""
        self.logger.info("Starting data processing pipeline")
        
        if self.load_from_hf:
            self.logger.info("Download from HuggingFace enabled")
            try:
                self.download_datasets()
            except Exception as e:
                self.logger.error(f"Dataset download failed: {e}")
                return
        else:
            self.logger.info("Using existing local data (HuggingFace download disabled)")
        
        self.logger.info("Setting up processors...")
        try:
            self.setup()
        except Exception as e:
            self.logger.error(f"Processor setup failed: {e}")
            return
        
        self.logger.info("Processing all datasets...")
        self.process_all()
        
        #remove data_raw folder and everything inside after processing
        data_raw_path = Path("data_raw/vcog-bench")
        if data_raw_path.exists() and data_raw_path.is_dir():
            try:
                shutil.rmtree(data_raw_path)
                self.logger.info("Removed temporary data_raw/vcog-bench directory after processing")
            except Exception as e:
                self.logger.error(f"Failed to remove data_raw/vcog-bench directory: {e}")

        self.logger.info("Pipeline execution complete!")