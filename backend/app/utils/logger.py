"""Structured JSON logging configuration."""
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data, default=str)


class LoggerAdapter:
    """Logger adapter that accepts extra keyword arguments for structured logging."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize the adapter with a logger."""
        self.logger = logger
    
    def _log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """Internal logging method that handles extra kwargs."""
        # Extract extra fields from kwargs
        extra_fields = {}
        # Common logging kwargs that should not be treated as extra fields
        reserved_kwargs = {'exc_info', 'stack_info', 'stacklevel', 'extra'}
        
        # Separate extra fields from reserved kwargs
        reserved_values = {}
        for key, value in list(kwargs.items()):
            if key in reserved_kwargs:
                reserved_values[key] = value
            else:
                extra_fields[key] = value
        
        # Add extra fields to the extra dict
        if extra_fields:
            if 'extra' not in reserved_values:
                reserved_values['extra'] = {}
            reserved_values['extra']['extra_fields'] = extra_fields
        
        # Only pass reserved kwargs to the standard logger
        # Use log() method instead of _log() for better compatibility
        self.logger.log(level, msg, *args, **reserved_values)
    
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical message."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an exception message."""
        kwargs.setdefault('exc_info', True)
        self._log(logging.ERROR, msg, *args, **kwargs)


def get_logger(name: str) -> LoggerAdapter:
    """Get a configured logger with JSON formatting that accepts extra kwargs."""
    base_logger = logging.getLogger(name)
    
    if not base_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)
        base_logger.propagate = False
    
    return LoggerAdapter(base_logger)


def log_with_context(logger: logging.Logger, level: int, message: str, **kwargs: Any) -> None:
    """Log a message with additional context fields."""
    extra = {"extra_fields": kwargs}
    logger.log(level, message, extra=extra)

