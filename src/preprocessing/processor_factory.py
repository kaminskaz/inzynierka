from src.preprocessing.base_processor import BaseProcessor
from src.technical.configs.dataset_config import DatasetConfig
from src.preprocessing.standard_processor import StandardProcessor
from src.preprocessing.bongard_processor import BongardProcessor


class ProcessorFactory:
    """Factory for creating appropriate processor instances."""

    @staticmethod
    def create_processor(
        config: DatasetConfig, sheet_maker=None
    ) -> BaseProcessor:
        """Create the appropriate processor based on dataset category."""
        if config.category == "BP":
            return BongardProcessor(config)
        elif config.category in ["standard", "choice_only"]:
            if sheet_maker is None:
                raise ValueError("sheet_maker is required for standard processors")
            return StandardProcessor(config, sheet_maker)
        else:
            raise ValueError(f"Unknown category: {config.category}")
