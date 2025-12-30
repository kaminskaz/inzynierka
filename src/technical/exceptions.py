class PipelineCriticalError(Exception):
    """Raised when the VLLM server or strategy encounters a fatal state."""
    pass