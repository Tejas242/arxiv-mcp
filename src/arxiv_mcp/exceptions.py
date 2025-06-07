"""Custom exceptions for the arXiv MCP server."""

from typing import Optional, Any, Dict


class ArxivMCPError(Exception):
    """Base exception for all arXiv MCP related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ArxivAPIError(ArxivMCPError):
    """Raised when arXiv API returns an error or unexpected response."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(message, details)


class ValidationError(ArxivMCPError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        self.field = field
        self.value = value
        details = {}
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)
        super().__init__(message, details)


class NetworkError(ArxivMCPError):
    """Raised when network operations fail."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.original_error = original_error
        details = {}
        if original_error:
            details['original_error'] = str(original_error)
            details['error_type'] = type(original_error).__name__
        super().__init__(message, details)


class ConfigurationError(ArxivMCPError):
    """Raised when there are configuration-related issues."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        self.config_key = config_key
        details = {}
        if config_key:
            details['config_key'] = config_key
        super().__init__(message, details)


class RateLimitError(ArxivAPIError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(self, message: str = "API rate limit exceeded", retry_after: Optional[int] = None):
        self.retry_after = retry_after
        details = {}
        if retry_after:
            details['retry_after'] = retry_after
        super().__init__(message, details=details)


class PaperNotFoundError(ArxivAPIError):
    """Raised when a requested paper is not found."""
    
    def __init__(self, arxiv_id: str):
        self.arxiv_id = arxiv_id
        message = f"Paper not found: {arxiv_id}"
        super().__init__(message, details={'arxiv_id': arxiv_id})


class InvalidArxivIdError(ValidationError):
    """Raised when an arXiv ID format is invalid."""
    
    def __init__(self, arxiv_id: str):
        self.arxiv_id = arxiv_id
        message = f"Invalid arXiv ID format: {arxiv_id}"
        super().__init__(message, field='arxiv_id', value=arxiv_id)


class SearchError(ArxivAPIError):
    """Raised when search operations fail."""
    
    def __init__(self, message: str, query: Optional[str] = None):
        self.query = query
        details = {}
        if query:
            details['query'] = query
        super().__init__(message, details=details)


class DownloadError(ArxivAPIError):
    """Raised when PDF download operations fail."""
    
    def __init__(self, message: str, arxiv_id: Optional[str] = None, url: Optional[str] = None):
        self.arxiv_id = arxiv_id
        self.url = url
        details = {}
        if arxiv_id:
            details['arxiv_id'] = arxiv_id
        if url:
            details['url'] = url
        super().__init__(message, details=details)


def format_error_for_user(error: Exception) -> str:
    """
    Format an error for user-friendly display.
    
    Args:
        error: The exception to format
        
    Returns:
        User-friendly error message
    """
    if isinstance(error, PaperNotFoundError):
        return f"Paper not found: {error.arxiv_id}. Please check the arXiv ID is correct."
    
    elif isinstance(error, InvalidArxivIdError):
        return (f"Invalid arXiv ID format: {error.arxiv_id}. "
                "Expected format: YYMM.NNNNN (e.g., 2301.00001) or old format like cs/0601001")
    
    elif isinstance(error, RateLimitError):
        retry_msg = f" Try again in {error.retry_after} seconds." if error.retry_after else ""
        return f"API rate limit exceeded.{retry_msg}"
    
    elif isinstance(error, ValidationError):
        field_msg = f" (field: {error.field})" if error.field else ""
        return f"Invalid input: {error.message}{field_msg}"
    
    elif isinstance(error, NetworkError):
        return f"Network error: {error.message}. Please check your internet connection."
    
    elif isinstance(error, ConfigurationError):
        return f"Configuration error: {error.message}"
    
    elif isinstance(error, ArxivAPIError):
        if error.status_code:
            return f"arXiv API error ({error.status_code}): {error.message}"
        return f"arXiv API error: {error.message}"
    
    elif isinstance(error, ArxivMCPError):
        return f"Error: {error.message}"
    
    else:
        return f"Unexpected error: {str(error)}"
