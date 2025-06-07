"""Logging configuration for the arXiv MCP server."""

import logging
import logging.handlers
import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

from .config import get_settings, LoggingConfig
from .exceptions import ConfigurationError


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add thread/process info if available
        if hasattr(record, 'thread') and record.thread:
            log_data['thread_id'] = record.thread
        if hasattr(record, 'process') and record.process:
            log_data['process_id'] = record.process
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                'filename', 'module', 'lineno', 'funcName', 'created', 
                'msecs', 'relativeCreated', 'thread', 'threadName', 
                'processName', 'process', 'exc_info', 'exc_text', 'stack_info'
            }:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Enhanced colored formatter for development."""
    
    def __init__(self):
        if HAS_COLORLOG:
            super().__init__()
            self.formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            super().__init__(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self.formatter = self
    
    def format(self, record: logging.LogRecord) -> str:
        return self.formatter.format(record)


class TextFormatter(logging.Formatter):
    """Standard text formatter."""
    
    def __init__(self):
        super().__init__(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        config: Optional logging configuration. If None, uses settings from config module.
    """
    if config is None:
        try:
            settings = get_settings()
            config = settings.logging
        except Exception as e:
            # Fallback to basic configuration
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load logging config, using defaults: {e}")
            return
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set log level
    try:
        log_level = getattr(logging, config.level.upper())
    except AttributeError:
        log_level = logging.INFO
        print(f"Warning: Invalid log level '{config.level}', using INFO")
    
    root_logger.setLevel(log_level)
    
    # Choose formatter
    if config.format == "json":
        formatter = JSONFormatter()
    elif config.format == "colored":
        formatter = ColoredFormatter()
    else:  # text
        formatter = TextFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if configured)
    if config.file_path:
        try:
            file_path = Path(config.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                filename=config.file_path,
                maxBytes=config.max_file_size,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
            
            # Use JSON formatter for file logs
            file_handler.setFormatter(JSONFormatter())
            root_logger.addHandler(file_handler)
            
        except Exception as e:
            # Log to console if file logging fails
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to setup file logging: {e}")
    
    # Setup logger hierarchy
    setup_logger_hierarchy(log_level)


def setup_logger_hierarchy(log_level: int) -> None:
    """Setup logger hierarchy with appropriate levels."""
    
    # Main application loggers
    logging.getLogger('arxiv_mcp').setLevel(log_level)
    
    # Third-party library loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('feedparser').setLevel(logging.WARNING)
    
    # Set more verbose logging for development
    try:
        settings = get_settings()
        if settings.server.debug:
            logging.getLogger('arxiv_mcp').setLevel(logging.DEBUG)
            logging.getLogger('httpx').setLevel(logging.INFO)
    except Exception:
        pass  # Ignore if settings not available


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_function_call(func_name: str, **kwargs) -> None:
    """
    Log a function call with parameters.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    logger = get_logger('arxiv_mcp.functions')
    
    # Filter sensitive information
    safe_kwargs = {}
    sensitive_keys = {'token', 'password', 'secret', 'key'}
    
    for key, value in kwargs.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            safe_kwargs[key] = '***REDACTED***'
        else:
            safe_kwargs[key] = value
    
    logger.debug(f"Function call: {func_name}", extra={
        'function_name': func_name,
        'parameters': safe_kwargs
    })


def log_function_call_decorator(logger_instance=None):
    """
    Decorator for automatic function call logging.
    
    Args:
        logger_instance: Optional logger instance to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_logger = logger_instance or get_logger('arxiv_mcp.functions')
            func_logger.debug(f"Calling async function: {func.__name__}")
            try:
                result = await func(*args, **kwargs)
                func_logger.debug(f"Function {func.__name__} completed successfully")
                return result
            except Exception as e:
                func_logger.error(f"Function {func.__name__} failed: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_logger = logger_instance or get_logger('arxiv_mcp.functions')
            func_logger.debug(f"Calling function: {func.__name__}")
            try:
                result = func(*args, **kwargs)
                func_logger.debug(f"Function {func.__name__} completed successfully")
                return result
            except Exception as e:
                func_logger.error(f"Function {func.__name__} failed: {e}")
                raise
        
        # Check if function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_api_request(url: str, method: str = 'GET', **kwargs) -> None:
    """
    Log an API request.
    
    Args:
        url: Request URL
        method: HTTP method
        **kwargs: Additional request details
    """
    logger = get_logger('arxiv_mcp.api')
    logger.debug(f"API request: {method} {url}", extra={
        'url': url,
        'method': method,
        'request_details': kwargs
    })


def log_api_response(url: str, status_code: int, response_time: float) -> None:
    """
    Log an API response.
    
    Args:
        url: Request URL
        status_code: HTTP status code
        response_time: Response time in seconds
    """
    logger = get_logger('arxiv_mcp.api')
    
    if status_code >= 400:
        logger.warning(f"API error response: {status_code} for {url}", extra={
            'url': url,
            'status_code': status_code,
            'response_time': response_time
        })
    else:
        logger.debug(f"API response: {status_code} for {url}", extra={
            'url': url,
            'status_code': status_code,
            'response_time': response_time
        })


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an error with context.
    
    Args:
        error: The exception that occurred
        context: Additional context information
    """
    logger = get_logger('arxiv_mcp.errors')
    
    error_data = {
        'error_type': type(error).__name__,
        'error_message': str(error),
    }
    
    if context:
        error_data['context'] = context
    
    logger.error(f"Error occurred: {error}", extra=error_data, exc_info=True)


def log_performance(operation: str, duration: float, **kwargs) -> None:
    """
    Log performance metrics.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        **kwargs: Additional metrics
    """
    logger = get_logger('arxiv_mcp.performance')
    
    metrics = {
        'operation': operation,
        'duration': duration,
        **kwargs
    }
    
    if duration > 5.0:  # Log slow operations as warnings
        logger.warning(f"Slow operation: {operation} took {duration:.2f}s", extra=metrics)
    else:
        logger.debug(f"Performance: {operation} took {duration:.2f}s", extra=metrics)


# Context manager for function logging
class LoggedFunction:
    """Context manager for automatic function logging."""
    
    def __init__(self, func_name: str, **kwargs):
        self.func_name = func_name
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        log_function_call(self.func_name, **self.kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            log_error(exc_val, context={
                'function': self.func_name,
                'parameters': self.kwargs,
                'duration': duration
            })
        else:
            log_performance(self.func_name, duration)


def logged_function(func_name: str, **kwargs):
    """
    Decorator/context manager for function logging.
    
    Args:
        func_name: Name of the function
        **kwargs: Function parameters
        
    Returns:
        LoggedFunction context manager
    """
    return LoggedFunction(func_name, **kwargs)


# Initialize logging on module import
try:
    setup_logging()
except Exception as e:
    # Fallback to basic logging if setup fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger(__name__).warning(f"Failed to setup advanced logging: {e}")
