"""Utility functions for the arXiv MCP server."""

import re
from typing import Optional
from urllib.parse import quote_plus


def format_arxiv_id(arxiv_id: str) -> str:
    """
    Format and validate an arXiv identifier.
    
    Args:
        arxiv_id: Raw arXiv identifier
        
    Returns:
        Formatted arXiv ID
        
    Raises:
        ValueError: If the arXiv ID format is invalid
    """
    # Remove any URL prefix if present
    if arxiv_id.startswith('http'):
        arxiv_id = arxiv_id.split('/')[-1]
    
    # Remove version suffix if present (e.g., v1, v2)
    arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
    
    # Validate format
    if not re.match(r'^\d{4}\.\d{4,5}$|^[a-z-]+/\d{7}$', arxiv_id):
        raise ValueError(f"Invalid arXiv ID format: {arxiv_id}")
    
    return arxiv_id


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
    """
    query_parts = []
    
    for field, value in kwargs.items():
        if value:
            if field in ['title', 'author', 'abstract', 'category', 'all']:
                # Handle multi-word queries
                if ' ' in str(value):
                    query_parts.append(f'{field}:"{value}"')
                else:
                    query_parts.append(f'{field}:{value}')
    
    return ' AND '.join(query_parts)


def format_paper_summary(paper) -> str:
    """
    Format a paper into a human-readable summary.
    
    Args:
        paper: ArxivPaper instance
        
    Returns:
        Formatted paper summary
    """
    authors = ', '.join([author.name for author in paper.authors[:3]])
    if len(paper.authors) > 3:
        authors += f' et al. ({len(paper.authors)} total authors)'
    
    summary = f"""**{paper.title}**

**Authors:** {authors}
**arXiv ID:** {paper.id}
**Published:** {paper.published.strftime('%Y-%m-%d')}
**Primary Category:** {paper.primary_category.term}

**Abstract:**
{paper.summary[:500]}{'...' if len(paper.summary) > 500 else ''}

**Links:**
- Abstract: {paper.abs_url}
- PDF: {paper.pdf_url}
"""
    
    if paper.comment:
        summary += f"\n**Comment:** {paper.comment}"
    
    if paper.journal_ref:
        summary += f"\n**Journal Reference:** {paper.journal_ref}"
    
    if paper.doi:
        summary += f"\n**DOI:** {paper.doi}"
    
    return summary


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
