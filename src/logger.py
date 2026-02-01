import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional

class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for structured logging.
    Useful for training metrics and machine-parsable logs.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        # Add basic exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields if they exist in valid JSON types
        if hasattr(record, "metrics"):
            log_obj["metrics"] = record.metrics # type: ignore
            
        return json.dumps(log_obj)

def configure_logging(level: int = logging.INFO, json_format: bool = False):
    """
    Configure the root logger.
    
    Args:
        level: Logging level (e.g., logging.INFO)
        json_format: If True, use JSON formatter. If False, use standard readable text.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

def log_metrics(logger: logging.Logger, metrics: Dict[str, Any], step: Optional[int] = None):
    """
    Helper to log a dictionary of metrics as a structured log event.
    
    Args:
        logger: Logger instance to use
        metrics: Dictionary of metric name -> value
        step: Optional training step/iteration number
    """
    if step is not None:
        metrics["step"] = step
        
    # Pass metrics via extra dict so JsonFormatter can pick it up
    # Note: This only works effectively if json_format=True is used
    logger.info("Training Metrics", extra={"metrics": metrics})
