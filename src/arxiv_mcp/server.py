"""arXiv MCP Server - Main server implementation with tools and resources."""

import asyncio
import logging
from typing import Any, List, Optional
from pathlib import Path
import tempfile
import os

from mcp.server.fastmcp import FastMCP
from mcp.types import Resource, TextResourceContents

from .arxiv_client import ArxivClient
from .models import SearchParams, ArxivPaper
from .utils import format_paper_summary, extract_arxiv_categories, format_arxiv_id, build_search_query

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("arxiv-mcp")

# Global client instance
arxiv_client: Optional[ArxivClient] = None


async def get_client() -> ArxivClient:
    """Get or create the arXiv client instance."""
    global arxiv_client
    if arxiv_client is None:
        arxiv_client = ArxivClient()
    return arxiv_client


@mcp.tool()
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
        client = await get_client()
        
        # Validate parameters
        max_results = max(1, min(max_results, 100))
        if sort_by not in ["relevance", "lastUpdatedDate", "submittedDate"]:
            sort_by = "relevance"
        if sort_order not in ["ascending", "descending"]:
            sort_order = "descending"
        
        params = SearchParams(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        result = await client.search(params)
        
        if not result.papers:
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
        
        if result.total_results > max_results:
            output += f"Note: Showing first {max_results} of {result.total_results} total results."
        
        return output
        
    except Exception as e:
        logger.error(f"Error searching papers: {e}")
        return f"Error searching papers: {str(e)}"


@mcp.tool()
async def get_paper_details(arxiv_id: str) -> str:
    """
    Get detailed information about a specific arXiv paper.
    
    Args:
        arxiv_id: The arXiv identifier (e.g., "2301.00001" or "cs.AI/0601001")
    
    Returns:
        Detailed paper information including abstract, authors, categories, etc.
    """
    try:
        client = await get_client()
        
        paper = await client.get_paper_by_id(arxiv_id)
        
        if paper is None:
            return f"Paper not found: {arxiv_id}"
        
        return format_paper_summary(paper)
        
    except ValueError as e:
        return f"Invalid arXiv ID format: {str(e)}"
    except Exception as e:
        logger.error(f"Error getting paper details: {e}")
        return f"Error getting paper details: {str(e)}"


@mcp.tool()
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
        client = await get_client()
        max_results = max(1, min(max_results, 50))
        
        result = await client.search_by_author(author_name, max_results)
        
        if not result.papers:
            return f"No papers found for author: '{author_name}'"
        
        output = f"Papers by {author_name} ({len(result.papers)} found):\n\n"
        
        for i, paper in enumerate(result.papers, 1):
            output += f"{i}. **{paper.title}** ({paper.id})\n"
            output += f"   Published: {paper.published.strftime('%Y-%m-%d')}\n"
            output += f"   Category: {paper.primary_category.term}\n"
            if paper.journal_ref:
                output += f"   Journal: {paper.journal_ref}\n"
            output += f"   PDF: https://arxiv.org/pdf/{paper.id}.pdf\n\n"
        
        return output
        
    except Exception as e:
        logger.error(f"Error searching by author: {e}")
        return f"Error searching by author: {str(e)}"


@mcp.tool()
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
        client = await get_client()
        max_results = max(1, min(max_results, 50))
        
        result = await client.search_by_category(category, max_results)
        
        if not result.papers:
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
        
        return output
        
    except Exception as e:
        logger.error(f"Error searching by category: {e}")
        return f"Error searching by category: {str(e)}"


@mcp.tool()
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
        client = await get_client()
        
        # Validate arXiv ID
        formatted_id = format_arxiv_id(arxiv_id)
        
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
        
        return f"PDF downloaded successfully to: {save_path}"
        
    except ValueError as e:
        return f"Invalid arXiv ID format: {str(e)}"
    except Exception as e:
        logger.error(f"Error downloading PDF: {e}")
        return f"Error downloading PDF: {str(e)}"


@mcp.tool()
async def get_arxiv_categories() -> str:
    """
    Get a list of available arXiv subject categories.
    
    Returns:
        List of arXiv categories with descriptions
    """
    try:
        categories = extract_arxiv_categories()
        
        output = "arXiv Subject Categories:\n\n"
        
        for code, description in categories.items():
            output += f"**{code}**: {description}\n"
        
        output += "\nNote: Many categories have subcategories (e.g., cs.AI, cs.LG, physics.atom-ph)"
        output += "\nUse the full category code when searching (e.g., 'cs.AI' not just 'cs')"
        
        return output
        
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return f"Error getting categories: {str(e)}"


@mcp.tool()
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
        
        if title_keywords:
            query_parts['title'] = title_keywords
        if author_name:
            query_parts['author'] = author_name
        if abstract_keywords:
            query_parts['abstract'] = abstract_keywords
        if category:
            query_parts['category'] = category
        if all_fields:
            query_parts['all'] = all_fields
        
        if not query_parts:
            return "Error: At least one search field must be provided"
        
        query = build_search_query(**query_parts)
        
        return f"Constructed query: {query}\n\nYou can now use this query with the search_papers tool."
        
    except Exception as e:
        logger.error(f"Error building query: {e}")
        return f"Error building query: {str(e)}"


# Resources for accessing recent papers by category
@mcp.resource("arxiv://recent/{category}")
async def get_recent_papers_resource(category: str) -> Resource:
    """
    Resource providing recent papers in a specific category.
    
    Args:
        category: arXiv category
    """
    try:
        client = await get_client()
        result = await client.search_by_category(category, 20)
        
        if not result.papers:
            content = f"No recent papers found in category: {category}"
        else:
            content = f"Recent papers in {category}:\n\n"
            for paper in result.papers:
                content += format_paper_summary(paper) + "\n" + "="*80 + "\n\n"
        
        return Resource(
            uri=f"arxiv://recent/{category}",
            name=f"Recent papers in {category}",
            description=f"Latest papers submitted to arXiv in the {category} category",
            mimeType="text/plain",
            contents=TextResourceContents(text=content)
        )
        
    except Exception as e:
        logger.error(f"Error getting recent papers resource: {e}")
        return Resource(
            uri=f"arxiv://recent/{category}",
            name=f"Error: Recent papers in {category}",
            description="Error retrieving recent papers",
            mimeType="text/plain",
            contents=TextResourceContents(text=f"Error: {str(e)}")
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
        client = await get_client()
        paper = await client.get_paper_by_id(paper_id)
        
        if paper is None:
            content = f"Paper not found: {paper_id}"
        else:
            content = format_paper_summary(paper)
        
        return Resource(
            uri=f"arxiv://paper/{paper_id}",
            name=f"arXiv Paper {paper_id}",
            description=f"Detailed information about arXiv paper {paper_id}",
            mimeType="text/plain",
            contents=TextResourceContents(text=content)
        )
        
    except Exception as e:
        logger.error(f"Error getting paper resource: {e}")
        return Resource(
            uri=f"arxiv://paper/{paper_id}",
            name=f"Error: arXiv Paper {paper_id}",
            description="Error retrieving paper information",
            mimeType="text/plain",
            contents=TextResourceContents(text=f"Error: {str(e)}")
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
        client = await get_client()
        paper = await client.get_paper_by_id(arxiv_id)
        
        if paper is None:
            return f"Paper not found: {arxiv_id}"
        
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
        
        return prompt
        
    except Exception as e:
        logger.error(f"Error generating summary prompt: {e}")
        return f"Error generating summary prompt: {str(e)}"


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
        client = await get_client()
        papers = []
        
        for arxiv_id in arxiv_ids:
            paper = await client.get_paper_by_id(arxiv_id)
            if paper:
                papers.append(paper)
        
        if len(papers) < 2:
            return "Error: Could not retrieve enough papers for comparison"
        
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
        
        return prompt
        
    except Exception as e:
        logger.error(f"Error generating comparison prompt: {e}")
        return f"Error generating comparison prompt: {str(e)}"


def main():
    """Main entry point for the arXiv MCP server."""
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()
