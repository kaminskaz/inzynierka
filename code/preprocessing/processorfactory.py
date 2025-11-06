from code.preprocessing.baseprocessor import BaseProcessor
from code.preprocessing.processorconfig import ProcessorConfig
from code.preprocessing.standardprocessor import StandardProcessor
from code.preprocessing.bongardprocessor import BongardProcessor

class ProcessorFactory:
    """Factory for creating appropriate processor instances."""
    
    @staticmethod
    def create_processor(
        dataset_name: str, 
        config: ProcessorConfig, 
        sheet_maker=None
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