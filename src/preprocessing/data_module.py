import os
import json
from pathlib import Path
import traceback
from typing import Dict, Tuple, Any
import PIL
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, login
from src.technical.configs.dataset_config import DatasetConfig
from src.preprocessing.processor_factory import ProcessorFactory
from src.technical.utils import get_config_path
import logging
import shutil


class DataModule:
    """Main data processing module."""

    def __init__(
        self,
        load_from_hf: bool = False,
    ):
        self.config_path = Path(get_config_path("dataset"))
        self.load_from_hf = load_from_hf
        self.logger = logging.getLogger("DataModule")
        self.configs: Dict[str, DatasetConfig] = {}
        self.raw_config_dicts: Dict[str, Dict[str, Any]] = {}
        self.configs, self.raw_config_dicts = self.load_configs()
        self.processors = {}
        load_dotenv()

    def load_configs(
        self,
    ) -> Tuple[Dict[str, DatasetConfig], Dict[str, Dict[str, Any]]]:
        """Load all dataset configurations."""
        try:
            with open(self.config_path, "r") as f:
                config_dict = json.load(f)

            configs = {}
            raw_cfgs = {}
            for name, cfg in config_dict.items():
                try:
                    configs[name] = DatasetConfig.from_dict(cfg)
                    raw_cfgs[name] = cfg  # Store the raw config dict
                except Exception as e:
                    self.logger.error(f"Error loading config for {name}: {e}")

            self.logger.info(f"Loaded {len(configs)} dataset configurations")
            return configs, raw_cfgs

        except Exception as e:
            self.logger.error(f"Error loading config file {self.config_path}: {e}")
            raise

    def setup(self) -> None:
        """Set up all processors."""
        # Import sheet maker (avoiding circular imports)
        from src.preprocessing.standard_sheetmaker import StandardSheetMaker

        sheet_maker = StandardSheetMaker()

        for dataset_name, config in self.configs.items():
            try:
                self.processors[dataset_name] = ProcessorFactory.create_processor(
                    config,
                    sheet_maker if config.category != "BP" else None,
                )
                self.logger.debug(f"Created processor for {dataset_name}")
            except Exception as e:
                self.logger.error(f"Error creating processor for {dataset_name}: {e}")

        self.logger.info(f"Set up {len(self.processors)} processors")

    def download_datasets(self) -> None:
        """Download datasets from HuggingFace."""
        # Only download if datasets have hf_repo_id specified
        datasets_to_download = {
            name: config for name, config in self.configs.items() if config.hf_repo_id
        }

        if not datasets_to_download:
            self.logger.info("No datasets configured for HuggingFace download")
            return

        self.logger.info(
            f"Found {len(datasets_to_download)} datasets to download from HuggingFace"
        )

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
        """Download or resume downloading a dataset from HuggingFace."""
        data_path = os.path.join("data_raw", repo_id.split("/")[-1])
        os.makedirs(data_path, exist_ok=True)

        self.logger.info(f"Checking existing data for {repo_id} in {data_path}...")

        try:
            # Attempt to download or resume — this handles incomplete downloads automatically
            self.logger.info(
                f"Starting (or resuming) download of {repo_id} to {data_path}"
            )
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=str(data_path),
                max_workers=1,
            )
            self.logger.info(f"Download (or resume) complete for {repo_id}")

        except Exception as e:
            self.logger.error(f"Error downloading {repo_id}: {e}", exc_info=True)
            raise

    def check_dataset_counts(self) -> bool:
        """
        Check actual downloaded sample counts against 'expected_num_samples' from config.
        
        Returns:
            bool: True if all counts match, False otherwise.
        """
        self.logger.info("Checking dataset sample counts...")
        all_match = True
        
        mismatches = []

        for dataset_name, config in self.configs.items():
            raw_cfg = self.raw_config_dicts[dataset_name]
            expected = raw_cfg.get("expected_num_samples")

            if expected is None:
                self.logger.debug(
                    f"No 'expected_num_samples' for {dataset_name}, skipping check."
                )
                continue

            raw_data_path = Path(config.data_folder)
            actual = 0

            try:
                if not raw_data_path.exists():
                    self.logger.warning(
                        f"Data folder for {dataset_name} not found: {raw_data_path}"
                    )
                else:
                    # Count subdirectories, as each represents a problem
                    actual = len(
                        [
                            p
                            for p in os.listdir(raw_data_path)
                            if os.path.isdir(os.path.join(raw_data_path, p))
                        ]
                    )
            except Exception as e:
                self.logger.error(
                    f"Error counting samples for {dataset_name} in {raw_data_path}: {e}"
                )

            if actual == expected:
                self.logger.info(
                    f"{dataset_name}: Found {actual} / {expected} samples (Match)"
                )
            else:
                msg = f"{dataset_name}: Found {actual} / {expected} samples (MISMATCH)"
                self.logger.error(msg)
                mismatches.append(msg)
                all_match = False

        if all_match:
            self.logger.info("All dataset counts match expected values.")
            return True
        else:
            self.logger.critical("DATASET VALIDATION FAILED")
            self.logger.critical("The following datasets have incomplete downloads:")
            for m in mismatches:
                self.logger.critical(f" - {m}")
            self.logger.critical("Please run the download again to fetch missing files.")
            return False

    def process_all(self) -> None:
        """Process all datasets."""
        total_start_time = logging.time if hasattr(logging, "time") else None

        for dataset_name, processor in self.processors.items():
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Processing dataset: {dataset_name}")
            self.logger.info(f"{'='*60}")

            try:
                processor.process()
            except Exception as e:
                self.logger.error(
                    f"Failed to process {dataset_name}: {e}", exc_info=True
                )

        # change all images to .png in data/ and folders inside
        for root, dirs, files in os.walk("data/"):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".bmp", ".gif", ".tiff")):
                    file_path = os.path.join(root, file)
                    img = PIL.Image.open(file_path)
                    png_file_path = os.path.splitext(file_path)[0] + ".png"
                    img.save(png_file_path, "PNG")
                    os.remove(file_path)

        self.logger.info(f"{'='*60}")
        self.logger.info("All datasets processing complete")
        self.logger.info(f"{'='*60}")

    def verify_outputs(self) -> None:
        """
        Checks all processed datasets for missing solutions or annotations
        by scanning the output 'data/' directory.
        """
        self.logger.info("Starting output verification...")

        for dataset_name, config_data in self.raw_config_dicts.items():
            self.logger.info(f"--- Checking: {dataset_name} ---")

            # 1. Get list of all processed problem IDs from the file system
            problem_dir = Path(os.path.join("data", dataset_name, "problems"))
            if not problem_dir.exists():
                self.logger.warning(f"  No 'problems' directory found for {dataset_name}.")
                continue

            try:
                processed_problems = set(
                    p.name for p in problem_dir.iterdir() if p.is_dir()
                )
                if not processed_problems:
                    self.logger.info(f"  No processed problems found for {dataset_name}.")
                    continue
                self.logger.info(f"  Found {len(processed_problems)} processed problems.")
            except Exception as e:
                self.logger.error(f"  Failed to list processed problems: {e}")
                continue

            json_dir = os.path.join("data", dataset_name, "jsons")
            solution_keys = set()
            annotation_keys = set()

            # 2. Load solutions
            solutions_path = os.path.join(json_dir, f"{dataset_name}_solutions.json")

            if os.path.exists(solutions_path):
                try:
                    with open(solutions_path, "r", encoding="utf-8") as f:
                        solutions_data = json.load(f)
                    solution_keys = set(solutions_data.keys())
                except Exception as e:
                    self.logger.error(f"  Failed to load solutions file {solutions_path}: {e}")
            else:
                self.logger.warning(f"  No solutions file found at {solutions_path}")

            # 3. Load annotations (if configured)
            has_annotations_config = bool(config_data.get("annotations_folder"))

            if has_annotations_config:
                annotations_path = os.path.join(json_dir, f"{dataset_name}_annotations.json")
                if os.path.exists(annotations_path):
                    try:
                        with open(annotations_path, "r", encoding="utf-8") as f:
                            annotations_data = json.load(f)
                        annotation_keys = set(annotations_data.keys())
                    except Exception as e:
                        self.logger.error(
                            f"  Failed to load annotations file {annotations_path}: {e}"
                        )
                else:
                    self.logger.warning(f"  No annotations file found at {annotations_path}")

            # 4. Compare and report
            # Solutions check
            missing_solutions = processed_problems - solution_keys
            if not os.path.exists(solutions_path):
                pass  # Already warned
            elif missing_solutions:
                self.logger.warning(
                    f"  Found {len(missing_solutions)} problems missing solutions."
                )
                if len(missing_solutions) < 10:
                    self.logger.warning(f"    Missing: {sorted(list(missing_solutions))}")
            else:
                self.logger.info(
                    f"  Solutions check passed (all {len(processed_problems)} problems have solutions)."
                )

            # Annotations check
            if not has_annotations_config:
                self.logger.info("  (Annotations not configured for this dataset)")
            else:
                missing_annotations = processed_problems - annotation_keys
                if not annotations_path.exists():
                    pass  
                elif missing_annotations:
                    self.logger.warning(
                        f"  Found {len(missing_annotations)} problems missing annotations."
                    )
                    if len(missing_annotations) < 10:
                        self.logger.warning(f"    Missing: {sorted(list(missing_annotations))}")
                else:
                    self.logger.info(
                        f"  Annotations check passed (all {len(processed_problems)} problems have annotations)."
                    )

        self.logger.info("--- Output check complete ---")

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
            self.logger.info(
                "Using existing local data (HuggingFace download disabled)"
            )

        self.logger.info("Checking downloaded dataset counts...")
        try:
            if not self.check_dataset_counts():
                self.logger.error(
                    "Pipeline stopped due to dataset count mismatches. "
                    "Please re-run with download enabled or fix the data manually."
                )
                return
        except Exception as e:
            self.logger.error(f"Failed to check dataset counts: {e}", exc_info=True)
            return 

        self.logger.info("Setting up processors...")
        try:
            self.setup()
        except Exception as e:
            self.logger.error(f"Processor setup failed: {e}")
            return

        self.logger.info("Processing all datasets...")
        self.process_all()

        self.logger.info("Verifying processed outputs...")
        self.verify_outputs()
        
        self.logger.info("Pipeline execution complete!") 