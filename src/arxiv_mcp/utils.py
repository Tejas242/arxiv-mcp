"""Utility functions for the arXiv MCP server."""

import re
from typing import Optional, Dict, Any
from urllib.parse import quote_plus

from .exceptions import ValidationError, InvalidArxivIdError
from .logging_config import get_logger

logger = get_logger(__name__)


def format_arxiv_id(arxiv_id: str) -> str:
    """
    Format and validate an arXiv identifier.
    
    Args:
        arxiv_id: Raw arXiv identifier
        
    Returns:
        Formatted arXiv ID
        
    Raises:
        InvalidArxivIdError: If the arXiv ID format is invalid
    """
    if not arxiv_id or not isinstance(arxiv_id, str):
        raise InvalidArxivIdError(str(arxiv_id))
    
    # Remove any URL prefix if present
    if arxiv_id.startswith('http'):
        arxiv_id = arxiv_id.split('/')[-1]
    
    # Remove version suffix if present (e.g., v1, v2)
    arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
    
    # Validate format
    new_format = re.match(r'^\d{4}\.\d{4,5}$', arxiv_id)
    old_format = re.match(r'^[a-z-]+/\d{7}$', arxiv_id)
    
    if not (new_format or old_format):
        logger.warning(f"Invalid arXiv ID format: {arxiv_id}")
        raise InvalidArxivIdError(arxiv_id)
    
    return arxiv_id


def validate_arxiv_id(arxiv_id: str) -> str:
    """
    Validate an arXiv identifier.
    
    Args:
        arxiv_id: arXiv identifier to validate
        
    Returns:
        Validated arXiv ID
        
    Raises:
        InvalidArxivIdError: If the arXiv ID format is invalid
    """
    if not arxiv_id or not isinstance(arxiv_id, str):
        raise InvalidArxivIdError(str(arxiv_id))
    
    # Use the existing format_arxiv_id function for validation
    return format_arxiv_id(arxiv_id)


def build_search_query(**kwargs) -> str:
    """
    Build an arXiv search query from keyword arguments.
    
    Supported fields:
    - title: Search in title
    - author: Search by author
    - abstract: Search in abstract
    - category: Search by category
    - all: Search all fields
    
    Args:
        **kwargs: Search field parameters
        
    Returns:
        Formatted search query string
        
    Raises:
        ValidationError: If invalid search parameters are provided
    """
    if not kwargs:
        return ""
    
    valid_fields = {'title', 'author', 'abstract', 'category', 'all'}
    query_parts = []
    
    for field, value in kwargs.items():
        if not value:
            continue
            
        if field not in valid_fields:
            logger.warning(f"Unknown search field: {field}")
            continue
        
        # Sanitize value
        if not isinstance(value, str):
            value = str(value)
        
        # Remove potentially harmful characters
        value = re.sub(r'[^\w\s\-\.\"\']', ' ', value).strip()
        
        if not value:
            continue
        
        # Handle multi-word queries
        if ' ' in value and not (value.startswith('"') and value.endswith('"')):
            query_parts.append(f'{field}:"{value}"')
        else:
            query_parts.append(f'{field}:{value}')
    
    if not query_parts:
        raise ValidationError("No valid search terms provided")
    
    return ' AND '.join(query_parts)


def format_paper_summary(paper) -> str:
    """
    Format a paper into a human-readable summary.
    
    Args:
        paper: ArxivPaper instance
        
    Returns:
        Formatted paper summary
        
    Raises:
        ValidationError: If paper object is invalid
    """
    try:
        if not paper:
            raise ValidationError("Paper object is None")
        
        if not hasattr(paper, 'title') or not paper.title:
            raise ValidationError("Paper missing title")
        
        # Safe author extraction
        authors = []
        if hasattr(paper, 'authors') and paper.authors:
            for author in paper.authors[:3]:
                if hasattr(author, 'name') and author.name:
                    authors.append(author.name)
        
        author_str = ', '.join(authors) if authors else "Unknown authors"
        if hasattr(paper, 'authors') and len(paper.authors) > 3:
            author_str += f' et al. ({len(paper.authors)} total authors)'
        
        # Safe date formatting
        published_str = "Unknown date"
        if hasattr(paper, 'published') and paper.published:
            try:
                published_str = paper.published.strftime('%Y-%m-%d')
            except Exception as e:
                logger.warning(f"Error formatting date: {e}")
                published_str = str(paper.published)
        
        # Safe category extraction
        category_str = "Unknown category"
        if hasattr(paper, 'primary_category') and paper.primary_category:
            if hasattr(paper.primary_category, 'term'):
                category_str = paper.primary_category.term
        
        # Safe summary truncation
        abstract = ""
        if hasattr(paper, 'summary') and paper.summary:
            abstract = paper.summary[:500]
            if len(paper.summary) > 500:
                abstract += '...'
        
        summary = f"""**{paper.title}**

**Authors:** {author_str}
**arXiv ID:** {getattr(paper, 'id', 'Unknown')}
**Published:** {published_str}
**Primary Category:** {category_str}

**Abstract:**
{abstract}

**Links:**
- Abstract: {getattr(paper, 'abs_url', 'N/A')}
- PDF: {getattr(paper, 'pdf_url', 'N/A')}
"""
        
        # Optional fields
        if hasattr(paper, 'comment') and paper.comment:
            summary += f"\n**Comment:** {paper.comment}"
        
        if hasattr(paper, 'journal_ref') and paper.journal_ref:
            summary += f"\n**Journal Reference:** {paper.journal_ref}"
        
        if hasattr(paper, 'doi') and paper.doi:
            summary += f"\n**DOI:** {paper.doi}"
        
        return summary
        
    except Exception as e:
        logger.error(f"Error formatting paper summary: {e}")
        raise ValidationError(f"Failed to format paper summary: {str(e)}")


def extract_arxiv_categories() -> dict:
    """
    Return a dictionary of arXiv subject categories and their descriptions.
    
    Returns:
        Dictionary mapping category codes to descriptions
    """
    return {
        # Physics
        'astro-ph': 'Astrophysics',
        'cond-mat': 'Condensed Matter',
        'gr-qc': 'General Relativity and Quantum Cosmology',
        'hep-ex': 'High Energy Physics - Experiment',
        'hep-lat': 'High Energy Physics - Lattice',
        'hep-ph': 'High Energy Physics - Phenomenology',
        'hep-th': 'High Energy Physics - Theory',
        'math-ph': 'Mathematical Physics',
        'nlin': 'Nonlinear Sciences',
        'nucl-ex': 'Nuclear Experiment',
        'nucl-th': 'Nuclear Theory',
        'physics': 'Physics',
        'quant-ph': 'Quantum Physics',
        
        # Mathematics
        'math': 'Mathematics',
        
        # Computer Science
        'cs': 'Computer Science',
        
        # Quantitative Biology
        'q-bio': 'Quantitative Biology',
        
        # Quantitative Finance
        'q-fin': 'Quantitative Finance',
        
        # Statistics
        'stat': 'Statistics',
        
        # Electrical Engineering and Systems Science
        'eess': 'Electrical Engineering and Systems Science',
        
        # Economics
        'econ': 'Economics'
    }


def validate_search_params(
    query: Optional[str] = None,
    max_results: Optional[int] = None,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate and sanitize search parameters.
    
    Args:
        query: Search query string
        max_results: Maximum number of results
        sort_by: Sort field
        sort_order: Sort direction
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ValidationError: If parameters are invalid
    """
    validated = {}
    
    # Validate query
    if query is not None:
        if not isinstance(query, str):
            raise ValidationError("Query must be a string", field="query", value=query)
        
        # Sanitize query
        query = query.strip()
        if not query:
            raise ValidationError("Query cannot be empty", field="query")
        
        # Check for reasonable length
        if len(query) > 1000:
            raise ValidationError("Query too long (max 1000 characters)", field="query")
        
        validated['query'] = query
    
    # Validate max_results
    if max_results is not None:
        if not isinstance(max_results, int):
            try:
                max_results = int(max_results)
            except (ValueError, TypeError):
                raise ValidationError("max_results must be an integer", field="max_results", value=max_results)
        
        if max_results < 1:
            raise ValidationError("max_results must be positive", field="max_results", value=max_results)
        
        if max_results > 100:
            logger.warning(f"Limiting max_results from {max_results} to 100")
            max_results = 100
        
        validated['max_results'] = max_results
    
    # Validate sort_by
    if sort_by is not None:
        if not isinstance(sort_by, str):
            raise ValidationError("sort_by must be a string", field="sort_by", value=sort_by)
        
        valid_sort_by = ["relevance", "lastUpdatedDate", "submittedDate"]
        if sort_by not in valid_sort_by:
            logger.warning(f"Invalid sort_by '{sort_by}', using 'relevance'")
            sort_by = "relevance"
        
        validated['sort_by'] = sort_by
    
    # Validate sort_order
    if sort_order is not None:
        if not isinstance(sort_order, str):
            raise ValidationError("sort_order must be a string", field="sort_order", value=sort_order)
        
        valid_sort_order = ["ascending", "descending"]
        if sort_order not in valid_sort_order:
            logger.warning(f"Invalid sort_order '{sort_order}', using 'descending'")
            sort_order = "descending"
        
        validated['sort_order'] = sort_order
    
    return validated


def sanitize_input(value: Any, max_length: int = 1000) -> str:
    """
    Sanitize user input by removing potentially harmful content.
    
    Args:
        value: Input value to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Raises:
        ValidationError: If input is invalid
    """
    if value is None:
        return ""
    
    if not isinstance(value, str):
        value = str(value)
    
    # Remove control characters
    value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
    
    # Remove excessive whitespace
    value = re.sub(r'\s+', ' ', value).strip()
    
    # Check length
    if len(value) > max_length:
        raise ValidationError(f"Input too long (max {max_length} characters)")
    
    return value


def sanitize_string(value: Any, max_length: int = 1000) -> str:
    """
    Sanitize a string input by removing potentially harmful content.
    
    Args:
        value: Input value to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Raises:
        ValidationError: If input is invalid
    """
    return sanitize_input(value, max_length)


def validate_arxiv_category(category: str) -> str:
    """
    Validate an arXiv category.
    
    Args:
        category: Category string to validate
        
    Returns:
        Validated category string
        
    Raises:
        ValidationError: If category is invalid
    """
    if not category or not isinstance(category, str):
        raise ValidationError("Category must be a non-empty string", field="category", value=category)
    
    category = category.strip().lower()
    
    # Basic category format validation
    if not re.match(r'^[a-z-]+(\.[a-z-]+)?$', category):
        raise ValidationError(f"Invalid category format: {category}", field="category", value=category)
    
    # Check against known categories
    known_categories = extract_arxiv_categories()
    main_category = category.split('.')[0]
    
    if main_category not in known_categories:
        logger.warning(f"Unknown category: {category}")
    
    return category


def safe_int_conversion(value: Any, default: int = 0, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """
    Safely convert a value to integer with bounds checking.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Converted integer value
    """
    try:
        result = int(value)
    except (ValueError, TypeError):
        logger.warning(f"Failed to convert '{value}' to int, using default {default}")
        return default
    
    if min_val is not None and result < min_val:
        logger.warning(f"Value {result} below minimum {min_val}, using minimum")
        return min_val
    
    if max_val is not None and result > max_val:
        logger.warning(f"Value {result} above maximum {max_val}, using maximum")
        return max_val
    
    return result
