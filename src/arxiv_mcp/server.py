"""arXiv MCP Server - Main server implementation with tools and resources."""

import asyncio
import tempfile
import os
from typing import Any, List, Optional
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import Resource, TextResourceContents

from .arxiv_client import ArxivClient
from .models import SearchParams, ArxivPaper
from .utils import (
    format_paper_summary, 
    extract_arxiv_categories, 
    format_arxiv_id, 
    build_search_query,
    validate_arxiv_id,
    validate_search_params,
    sanitize_string
)
from .config import get_config
from .logging_config import setup_logging, get_logger
from .exceptions import (
    ArxivMCPError,
    ValidationError,
    InvalidArxivIdError,
    PaperNotFoundError,
    ArxivAPIError,
    NetworkError
)

# Initialize configuration and logging
config = get_config()
setup_logging(config.logging)
logger = get_logger(__name__)

# Import log_function_call after logging is set up
try:
    from .logging_config import log_function_call_decorator as log_function_call
except ImportError:
    # Fallback if import fails
    def log_function_call(logger_instance=None):
        def decorator(func):
            return func
        return decorator

# Initialize FastMCP server
mcp = FastMCP("arxiv-mcp")

# Global client instance
arxiv_client: Optional[ArxivClient] = None


async def get_client() -> ArxivClient:
    """Get or create the arXiv client instance."""
    global arxiv_client
    if arxiv_client is None:
        arxiv_client = ArxivClient(config.arxiv_api)
    return arxiv_client


@mcp.tool()
@log_function_call(logger)
async def search_papers(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    sort_order: str = "descending"
) -> str:
    """
    Search for academic papers on arXiv.
    
    Args:
        query: Search query. Can use arXiv query syntax (e.g., "ti:machine learning", "au:Smith", "cat:cs.AI")
        max_results: Maximum number of results to return (1-100)
        sort_by: Sort by 'relevance', 'lastUpdatedDate', or 'submittedDate'
        sort_order: Sort order 'ascending' or 'descending'
    
    Returns:
        Formatted list of papers matching the search criteria
    """
    try:
        # Sanitize and validate inputs
        query = sanitize_string(query)
        if not query.strip():
            raise ValidationError("Query cannot be empty")
        
        # Validate search parameters
        validated_params = validate_search_params(
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        client = await get_client()
        
        params = SearchParams(
            query=query,
            max_results=validated_params['max_results'],
            sort_by=validated_params['sort_by'],
            sort_order=validated_params['sort_order']
        )
        
        logger.info(f"Searching papers with query: '{query}', max_results: {params.max_results}")
        result = await client.search(params)
        
        if not result.papers:
            logger.info(f"No papers found for query: '{query}'")
            return f"No papers found for query: '{query}'"
        
        # Format results
        output = f"Found {result.total_results} papers (showing {len(result.papers)}):\n\n"
        
        for i, paper in enumerate(result.papers, 1):
            authors = ', '.join([author.name for author in paper.authors[:3]])
            if len(paper.authors) > 3:
                authors += f" et al."
            
            output += f"{i}. **{paper.title}**\n"
            output += f"   Authors: {authors}\n"
            output += f"   arXiv ID: {paper.id}\n"
            output += f"   Category: {paper.primary_category.term}\n"
            output += f"   Published: {paper.published.strftime('%Y-%m-%d')}\n"
            output += f"   Abstract: {paper.summary[:200]}...\n"
            output += f"   PDF: https://arxiv.org/pdf/{paper.id}.pdf\n\n"
        
        if result.total_results > validated_params['max_results']:
            output += f"Note: Showing first {validated_params['max_results']} of {result.total_results} total results."
        
        logger.info(f"Successfully returned {len(result.papers)} papers")
        return output
        
    except ValidationError as e:
        logger.warning(f"Validation error in search_papers: {e}")
        return f"Validation error: {e}"
    except ArxivAPIError as e:
        logger.error(f"arXiv API error in search_papers: {e}")
        return f"arXiv API error: {e}"
    except NetworkError as e:
        logger.error(f"Network error in search_papers: {e}")
        return f"Network error: {e}. Please check your internet connection and try again."
    except ArxivMCPError as e:
        logger.error(f"arXiv MCP error in search_papers: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in search_papers: {e}", exc_info=True)
        return f"An unexpected error occurred. Please try again later."


@mcp.tool()
@log_function_call(logger)
async def get_paper_details(arxiv_id: str) -> str:
    """
    Get detailed information about a specific arXiv paper.
    
    Args:
        arxiv_id: The arXiv identifier (e.g., "2301.00001" or "cs.AI/0601001")
    
    Returns:
        Detailed paper information including abstract, authors, categories, etc.
    """
    try:
        # Validate and sanitize arXiv ID
        arxiv_id = sanitize_string(arxiv_id)
        validate_arxiv_id(arxiv_id)
        
        client = await get_client()
        
        logger.info(f"Fetching paper details for arXiv ID: {arxiv_id}")
        paper = await client.get_paper_by_id(arxiv_id)
        
        if paper is None:
            logger.warning(f"Paper not found: {arxiv_id}")
            raise PaperNotFoundError(arxiv_id)
        
        logger.info(f"Successfully retrieved paper: {paper.title}")
        return format_paper_summary(paper)
        
    except InvalidArxivIdError as e:
        logger.warning(f"Invalid arXiv ID: {e}")
        return f"Invalid arXiv ID format: {e}"
    except PaperNotFoundError as e:
        logger.warning(f"Paper not found: {e}")
        return str(e)
    except ArxivAPIError as e:
        logger.error(f"arXiv API error in get_paper_details: {e}")
        return f"arXiv API error: {e}"
    except NetworkError as e:
        logger.error(f"Network error in get_paper_details: {e}")
        return f"Network error: {e}. Please check your internet connection and try again."
    except ArxivMCPError as e:
        logger.error(f"arXiv MCP error in get_paper_details: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in get_paper_details: {e}", exc_info=True)
        return f"An unexpected error occurred. Please try again later."


@mcp.tool()
@log_function_call(logger)
async def search_by_author(author_name: str, max_results: int = 10) -> str:
    """
    Search for papers by a specific author.
    
    Args:
        author_name: Name of the author to search for
        max_results: Maximum number of results to return (1-50)
    
    Returns:
        List of papers by the specified author
    """
    try:
        # Sanitize and validate inputs
        author_name = sanitize_string(author_name)
        if not author_name.strip():
            raise ValidationError("Author name cannot be empty")
        
        # Validate max_results
        max_results = max(1, min(max_results, 50))
        
        client = await get_client()
        
        logger.info(f"Searching papers by author: '{author_name}', max_results: {max_results}")
        result = await client.search_by_author(author_name, max_results)
        
        if not result.papers:
            logger.info(f"No papers found for author: '{author_name}'")
            return f"No papers found for author: '{author_name}'"
        
        output = f"Papers by {author_name} ({len(result.papers)} found):\n\n"
        
        for i, paper in enumerate(result.papers, 1):
            output += f"{i}. **{paper.title}** ({paper.id})\n"
            output += f"   Published: {paper.published.strftime('%Y-%m-%d')}\n"
            output += f"   Category: {paper.primary_category.term}\n"
            if paper.journal_ref:
                output += f"   Journal: {paper.journal_ref}\n"
            output += f"   PDF: https://arxiv.org/pdf/{paper.id}.pdf\n\n"
        
        logger.info(f"Successfully returned {len(result.papers)} papers by author")
        return output
        
    except ValidationError as e:
        logger.warning(f"Validation error in search_by_author: {e}")
        return f"Validation error: {e}"
    except ArxivAPIError as e:
        logger.error(f"arXiv API error in search_by_author: {e}")
        return f"arXiv API error: {e}"
    except NetworkError as e:
        logger.error(f"Network error in search_by_author: {e}")
        return f"Network error: {e}. Please check your internet connection and try again."
    except ArxivMCPError as e:
        logger.error(f"arXiv MCP error in search_by_author: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in search_by_author: {e}", exc_info=True)
        return f"An unexpected error occurred. Please try again later."


@mcp.tool()
@log_function_call(logger)
async def search_by_category(category: str, max_results: int = 10) -> str:
    """
    Search for recent papers in a specific arXiv category.
    
    Args:
        category: arXiv category (e.g., 'cs.AI', 'physics.gen-ph', 'math.AG')
        max_results: Maximum number of results to return (1-50)
    
    Returns:
        List of recent papers in the specified category
    """
    try:
        # Sanitize and validate inputs
        category = sanitize_string(category)
        if not category.strip():
            raise ValidationError("Category cannot be empty")
        
        # Validate max_results
        max_results = max(1, min(max_results, 50))
        
        client = await get_client()
        
        logger.info(f"Searching papers in category: '{category}', max_results: {max_results}")
        result = await client.search_by_category(category, max_results)
        
        if not result.papers:
            logger.info(f"No papers found in category: '{category}'")
            return f"No papers found in category: '{category}'"
        
        output = f"Recent papers in {category} ({len(result.papers)} found):\n\n"
        
        for i, paper in enumerate(result.papers, 1):
            authors = ', '.join([author.name for author in paper.authors[:2]])
            if len(paper.authors) > 2:
                authors += f" et al."
            
            output += f"{i}. **{paper.title}**\n"
            output += f"   Authors: {authors}\n"
            output += f"   arXiv ID: {paper.id}\n"
            output += f"   Published: {paper.published.strftime('%Y-%m-%d')}\n"
            output += f"   Abstract: {paper.summary[:150]}...\n\n"
        
        logger.info(f"Successfully returned {len(result.papers)} papers in category")
        return output
        
    except ValidationError as e:
        logger.warning(f"Validation error in search_by_category: {e}")
        return f"Validation error: {e}"
    except ArxivAPIError as e:
        logger.error(f"arXiv API error in search_by_category: {e}")
        return f"arXiv API error: {e}"
    except NetworkError as e:
        logger.error(f"Network error in search_by_category: {e}")
        return f"Network error: {e}. Please check your internet connection and try again."
    except ArxivMCPError as e:
        logger.error(f"arXiv MCP error in search_by_category: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in search_by_category: {e}", exc_info=True)
        return f"An unexpected error occurred. Please try again later."


@mcp.tool()
@log_function_call(logger)
async def download_paper_pdf(arxiv_id: str, save_path: Optional[str] = None) -> str:
    """
    Download the PDF of an arXiv paper.
    
    Args:
        arxiv_id: The arXiv identifier
        save_path: Optional path to save the PDF (if not provided, saves to temp directory)
    
    Returns:
        Path to the downloaded PDF file
    """
    try:
        # Validate and sanitize arXiv ID
        arxiv_id = sanitize_string(arxiv_id)
        validate_arxiv_id(arxiv_id)
        
        # Validate save_path if provided
        if save_path:
            save_path = sanitize_string(save_path)
        
        client = await get_client()
        
        # Format arXiv ID
        formatted_id = format_arxiv_id(arxiv_id)
        
        logger.info(f"Downloading PDF for arXiv ID: {formatted_id}")
        
        # Download PDF content
        pdf_content = await client.download_pdf(formatted_id)
        
        # Determine save path
        if save_path is None:
            temp_dir = tempfile.gettempdir()
            save_path = os.path.join(temp_dir, f"{formatted_id.replace('/', '_')}.pdf")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save PDF
        with open(save_path, 'wb') as f:
            f.write(pdf_content)
        
        logger.info(f"PDF successfully downloaded to: {save_path}")
        return f"PDF downloaded successfully to: {save_path}"
        
    except InvalidArxivIdError as e:
        logger.warning(f"Invalid arXiv ID for PDF download: {e}")
        return f"Invalid arXiv ID format: {e}"
    except PaperNotFoundError as e:
        logger.warning(f"Paper not found for PDF download: {e}")
        return str(e)
    except ArxivAPIError as e:
        logger.error(f"arXiv API error in download_paper_pdf: {e}")
        return f"arXiv API error: {e}"
    except NetworkError as e:
        logger.error(f"Network error in download_paper_pdf: {e}")
        return f"Network error: {e}. Please check your internet connection and try again."
    except PermissionError as e:
        logger.error(f"Permission error saving PDF: {e}")
        return f"Permission error: Cannot save PDF to the specified location. Please check file permissions."
    except OSError as e:
        logger.error(f"OS error saving PDF: {e}")
        return f"File system error: {e}"
    except ArxivMCPError as e:
        logger.error(f"arXiv MCP error in download_paper_pdf: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in download_paper_pdf: {e}", exc_info=True)
        return f"An unexpected error occurred. Please try again later."


@mcp.tool()
@log_function_call(logger)
async def get_arxiv_categories() -> str:
    """
    Get a list of available arXiv subject categories.
    
    Returns:
        List of arXiv categories with descriptions
    """
    try:
        logger.info("Retrieving arXiv categories")
        categories = extract_arxiv_categories()
        
        output = "arXiv Subject Categories:\n\n"
        
        for code, description in categories.items():
            output += f"**{code}**: {description}\n"
        
        output += "\nNote: Many categories have subcategories (e.g., cs.AI, cs.LG, physics.atom-ph)"
        output += "\nUse the full category code when searching (e.g., 'cs.AI' not just 'cs')"
        
        logger.info(f"Successfully returned {len(categories)} categories")
        return output
        
    except ArxivMCPError as e:
        logger.error(f"arXiv MCP error in get_arxiv_categories: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in get_arxiv_categories: {e}", exc_info=True)
        return f"An unexpected error occurred. Please try again later."


@mcp.tool()
@log_function_call(logger)
async def build_advanced_query(
    title_keywords: Optional[str] = None,
    author_name: Optional[str] = None,
    abstract_keywords: Optional[str] = None,
    category: Optional[str] = None,
    all_fields: Optional[str] = None
) -> str:
    """
    Build an advanced search query using multiple fields.
    
    Args:
        title_keywords: Keywords to search in paper titles
        author_name: Author name to search for
        abstract_keywords: Keywords to search in abstracts
        category: arXiv category to filter by
        all_fields: Keywords to search across all fields
    
    Returns:
        The constructed query string that can be used with search_papers
    """
    try:
        query_parts = {}
        
        # Sanitize and validate inputs
        if title_keywords:
            title_keywords = sanitize_string(title_keywords)
            if title_keywords.strip():
                query_parts['title'] = title_keywords
        
        if author_name:
            author_name = sanitize_string(author_name)
            if author_name.strip():
                query_parts['author'] = author_name
        
        if abstract_keywords:
            abstract_keywords = sanitize_string(abstract_keywords)
            if abstract_keywords.strip():
                query_parts['abstract'] = abstract_keywords
        
        if category:
            category = sanitize_string(category)
            if category.strip():
                query_parts['category'] = category
        
        if all_fields:
            all_fields = sanitize_string(all_fields)
            if all_fields.strip():
                query_parts['all'] = all_fields
        
        if not query_parts:
            raise ValidationError("At least one search field must be provided")
        
        logger.info(f"Building advanced query with fields: {list(query_parts.keys())}")
        query = build_search_query(**query_parts)
        
        logger.info(f"Successfully built query: {query}")
        return f"Constructed query: {query}\n\nYou can now use this query with the search_papers tool."
        
    except ValidationError as e:
        logger.warning(f"Validation error in build_advanced_query: {e}")
        return f"Validation error: {e}"
    except ArxivMCPError as e:
        logger.error(f"arXiv MCP error in build_advanced_query: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in build_advanced_query: {e}", exc_info=True)
        return f"An unexpected error occurred. Please try again later."


# Resources for accessing recent papers by category
@mcp.resource("arxiv://recent/{category}")
async def get_recent_papers_resource(category: str) -> Resource:
    """
    Resource providing recent papers in a specific category.
    
    Args:
        category: arXiv category
    """
    try:
        # Sanitize category
        category = sanitize_string(category)
        
        logger.info(f"Getting recent papers resource for category: {category}")
        client = await get_client()
        result = await client.search_by_category(category, 20)
        
        if not result.papers:
            content = f"No recent papers found in category: {category}"
            logger.info(f"No recent papers found in category: {category}")
        else:
            content = f"Recent papers in {category}:\n\n"
            for paper in result.papers:
                content += format_paper_summary(paper) + "\n" + "="*80 + "\n\n"
            logger.info(f"Successfully retrieved {len(result.papers)} recent papers")
        
        return Resource(
            uri=f"arxiv://recent/{category}",
            name=f"Recent papers in {category}",
            description=f"Latest papers submitted to arXiv in the {category} category",
            mimeType="text/plain",
            contents=TextResourceContents(text=content)
        )
        
    except ArxivAPIError as e:
        logger.error(f"arXiv API error in get_recent_papers_resource: {e}")
        error_content = f"arXiv API error: {e}"
    except NetworkError as e:
        logger.error(f"Network error in get_recent_papers_resource: {e}")
        error_content = f"Network error: {e}. Please check your internet connection and try again."
    except ArxivMCPError as e:
        logger.error(f"arXiv MCP error in get_recent_papers_resource: {e}")
        error_content = f"Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in get_recent_papers_resource: {e}", exc_info=True)
        error_content = f"An unexpected error occurred: {e}"
    
    return Resource(
        uri=f"arxiv://recent/{category}",
        name=f"Error: Recent papers in {category}",
        description="Error retrieving recent papers",
        mimeType="text/plain",
        contents=TextResourceContents(text=error_content)
    )


# Resource for paper details
@mcp.resource("arxiv://paper/{paper_id}")
async def get_paper_resource(paper_id: str) -> Resource:
    """
    Resource providing detailed information about a specific paper.
    
    Args:
        paper_id: arXiv paper identifier
    """
    try:
        # Sanitize and validate paper ID
        paper_id = sanitize_string(paper_id)
        validate_arxiv_id(paper_id)
        
        logger.info(f"Getting paper resource for ID: {paper_id}")
        client = await get_client()
        paper = await client.get_paper_by_id(paper_id)
        
        if paper is None:
            content = f"Paper not found: {paper_id}"
            logger.warning(f"Paper not found: {paper_id}")
        else:
            content = format_paper_summary(paper)
            logger.info(f"Successfully retrieved paper resource: {paper.title}")
        
        return Resource(
            uri=f"arxiv://paper/{paper_id}",
            name=f"arXiv Paper {paper_id}",
            description=f"Detailed information about arXiv paper {paper_id}",
            mimeType="text/plain",
            contents=TextResourceContents(text=content)
        )
        
    except InvalidArxivIdError as e:
        logger.warning(f"Invalid arXiv ID for paper resource: {e}")
        error_content = f"Invalid arXiv ID format: {e}"
    except PaperNotFoundError as e:
        logger.warning(f"Paper not found for resource: {e}")
        error_content = str(e)
    except ArxivAPIError as e:
        logger.error(f"arXiv API error in get_paper_resource: {e}")
        error_content = f"arXiv API error: {e}"
    except NetworkError as e:
        logger.error(f"Network error in get_paper_resource: {e}")
        error_content = f"Network error: {e}. Please check your internet connection and try again."
    except ArxivMCPError as e:
        logger.error(f"arXiv MCP error in get_paper_resource: {e}")
        error_content = f"Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in get_paper_resource: {e}", exc_info=True)
        error_content = f"An unexpected error occurred: {e}"
    
    return Resource(
        uri=f"arxiv://paper/{paper_id}",
        name=f"Error: arXiv Paper {paper_id}",
        description="Error retrieving paper information",
        mimeType="text/plain",
        contents=TextResourceContents(text=error_content)
    )


# Prompts for common research tasks
@mcp.prompt()
async def summarize_paper(arxiv_id: str) -> str:
    """
    Generate a prompt for summarizing an arXiv paper.
    
    Args:
        arxiv_id: The arXiv identifier of the paper to summarize
    """
    try:
        # Sanitize and validate arXiv ID
        arxiv_id = sanitize_string(arxiv_id)
        validate_arxiv_id(arxiv_id)
        
        logger.info(f"Generating summary prompt for arXiv ID: {arxiv_id}")
        client = await get_client()
        paper = await client.get_paper_by_id(arxiv_id)
        
        if paper is None:
            logger.warning(f"Paper not found for summary prompt: {arxiv_id}")
            raise PaperNotFoundError(arxiv_id)
        
        prompt = f"""Please provide a comprehensive summary of this arXiv paper:

Title: {paper.title}
Authors: {', '.join([author.name for author in paper.authors])}
arXiv ID: {paper.id}
Category: {paper.primary_category.term}
Published: {paper.published.strftime('%Y-%m-%d')}

Abstract:
{paper.summary}

Please summarize this paper covering:
1. Main research question/problem addressed
2. Key methodology or approach used
3. Main findings/results
4. Significance and potential impact
5. Limitations or future work mentioned

Keep the summary concise but comprehensive, suitable for researchers in related fields."""
        
        logger.info(f"Successfully generated summary prompt for: {paper.title}")
        return prompt
        
    except InvalidArxivIdError as e:
        logger.warning(f"Invalid arXiv ID for summary prompt: {e}")
        return f"Invalid arXiv ID format: {e}"
    except PaperNotFoundError as e:
        logger.warning(f"Paper not found for summary prompt: {e}")
        return str(e)
    except ArxivAPIError as e:
        logger.error(f"arXiv API error in summarize_paper: {e}")
        return f"arXiv API error: {e}"
    except NetworkError as e:
        logger.error(f"Network error in summarize_paper: {e}")
        return f"Network error: {e}. Please check your internet connection and try again."
    except ArxivMCPError as e:
        logger.error(f"arXiv MCP error in summarize_paper: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in summarize_paper: {e}", exc_info=True)
        return f"An unexpected error occurred. Please try again later."


@mcp.prompt()
async def compare_papers(*arxiv_ids: str) -> str:
    """
    Generate a prompt for comparing multiple arXiv papers.
    
    Args:
        *arxiv_ids: Multiple arXiv identifiers to compare
    """
    if len(arxiv_ids) < 2:
        return "Error: At least two paper IDs are required for comparison"
    
    try:
        # Sanitize and validate all arXiv IDs
        sanitized_ids = []
        for arxiv_id in arxiv_ids:
            sanitized_id = sanitize_string(arxiv_id)
            validate_arxiv_id(sanitized_id)
            sanitized_ids.append(sanitized_id)
        
        logger.info(f"Generating comparison prompt for {len(sanitized_ids)} papers")
        client = await get_client()
        papers = []
        
        for arxiv_id in sanitized_ids:
            paper = await client.get_paper_by_id(arxiv_id)
            if paper:
                papers.append(paper)
            else:
                logger.warning(f"Paper not found: {arxiv_id}")
        
        if len(papers) < 2:
            raise ValidationError("Could not retrieve enough papers for comparison")
        
        prompt = "Please compare and contrast the following arXiv papers:\n\n"
        
        for i, paper in enumerate(papers, 1):
            prompt += f"Paper {i}:\n"
            prompt += f"Title: {paper.title}\n"
            prompt += f"Authors: {', '.join([author.name for author in paper.authors])}\n"
            prompt += f"arXiv ID: {paper.id}\n"
            prompt += f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
            prompt += f"Abstract: {paper.summary}\n\n"
        
        prompt += """Please provide a detailed comparison covering:
1. Research objectives and questions addressed by each paper
2. Methodologies and approaches used
3. Key findings and contributions
4. Similarities and differences in approach
5. Complementary aspects or contradictions
6. Which paper makes stronger contributions and why
7. How these works relate to the broader field

Organize your analysis in a clear, structured format."""
        
        logger.info(f"Successfully generated comparison prompt for {len(papers)} papers")
        return prompt
        
    except InvalidArxivIdError as e:
        logger.warning(f"Invalid arXiv ID for comparison prompt: {e}")
        return f"Invalid arXiv ID format: {e}"
    except ValidationError as e:
        logger.warning(f"Validation error in compare_papers: {e}")
        return f"Validation error: {e}"
    except ArxivAPIError as e:
        logger.error(f"arXiv API error in compare_papers: {e}")
        return f"arXiv API error: {e}"
    except NetworkError as e:
        logger.error(f"Network error in compare_papers: {e}")
        return f"Network error: {e}. Please check your internet connection and try again."
    except ArxivMCPError as e:
        logger.error(f"arXiv MCP error in compare_papers: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in compare_papers: {e}", exc_info=True)
        return f"An unexpected error occurred. Please try again later."


def main():
    """Main entry point for the arXiv MCP server."""
    try:
        logger.info("Starting arXiv MCP server")
        logger.info(f"Configuration loaded: {config.model_dump()}")
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
