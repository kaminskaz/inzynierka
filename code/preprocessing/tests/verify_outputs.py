import json
import logging
from pathlib import Path
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OutputVerifier")


def check_processing_outputs(
    config_path: str = "code/technical/configs/dataset_config.json",
):
    """
    Checks all processed datasets for missing solutions or annotations
    by reading the config file and scanning the output 'data/' directory.
    """
    logger.info("Starting output verification...")

    try:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        return

    logger.info(f"Loaded {len(config_dict)} dataset configurations.")

    for dataset_name, config_data in config_dict.items():
        logger.info(f"--- Checking: {dataset_name} ---")

        # 1. Get list of all processed problem IDs from the file system
        problem_dir = Path("data") / dataset_name / "problems"
        if not problem_dir.exists():
            logger.warning(f"  No 'problems' directory found for {dataset_name}.")
            continue

        try:
            processed_problems = set(
                p.name for p in problem_dir.iterdir() if p.is_dir()
            )
            if not processed_problems:
                logger.info(f"  No processed problems found for {dataset_name}.")
                continue
            logger.info(f"  Found {len(processed_problems)} processed problems.")
        except Exception as e:
            logger.error(f"  Failed to list processed problems: {e}")
            continue

        json_dir = Path("data") / dataset_name / "jsons"
        solution_keys = set()
        annotation_keys = set()

        # 2. Load solutions
        # Handle special case for 'BP'
        if dataset_name == "BP":
            solutions_path = json_dir / "bp_solutions.json"
        else:
            solutions_path = json_dir / f"{dataset_name}_solutions.json"

        if solutions_path.exists():
            try:
                with open(solutions_path, "r", encoding="utf-8") as f:
                    solutions_data = json.load(f)
                solution_keys = set(solutions_data.keys())
            except Exception as e:
                logger.error(f"  Failed to load solutions file {solutions_path}: {e}")
        else:
            logger.warning(f"  No solutions file found at {solutions_path}")

        # 3. Load annotations (if configured)
        # We read directly from the raw config dict
        has_annotations_config = bool(config_data.get("annotations_folder"))

        if has_annotations_config:
            annotations_path = json_dir / f"{dataset_name}_annotations.json"
            if annotations_path.exists():
                try:
                    with open(annotations_path, "r", encoding="utf-8") as f:
                        annotations_data = json.load(f)
                    annotation_keys = set(annotations_data.keys())
                except Exception as e:
                    logger.error(
                        f"  Failed to load annotations file {annotations_path}: {e}"
                    )
            else:
                logger.warning(f"  No annotations file found at {annotations_path}")

        # 4. Compare and report
        # Solutions check
        missing_solutions = processed_problems - solution_keys
        if not solutions_path.exists():
            pass  # Already warned
        elif missing_solutions:
            logger.warning(
                f"  Found {len(missing_solutions)} problems missing solutions."
            )
            if len(missing_solutions) < 10:
                logger.warning(f"     Missing: {sorted(list(missing_solutions))}")
        else:
            logger.info(
                f"  Solutions check passed (all {len(processed_problems)} problems have solutions)."
            )

        # Annotations check
        if not has_annotations_config:
            logger.info("  (Annotations not configured for this dataset)")
        else:
            missing_annotations = processed_problems - annotation_keys
            if not annotations_path.exists():
                pass  
            elif missing_annotations:
                logger.warning(
                    f"  Found {len(missing_annotations)} problems missing annotations."
                )
                if len(missing_annotations) < 10:
                    logger.warning(f"     Missing: {sorted(list(missing_annotations))}")
            else:
                logger.info(
                    f"  Annotations check passed (all {len(processed_problems)} problems have annotations)."
                )

    logger.info("--- Output check complete ---")


if __name__ == "__main__":
    # You can customize the path if it's different
    DEFAULT_CONFIG_PATH = "code/technical/configs/dataset_config.json"
    check_processing_outputs(DEFAULT_CONFIG_PATH)
