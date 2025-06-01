"""arXiv API client for retrieving papers and metadata."""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urlencode

import httpx
import feedparser
from dateutil.parser import parse as parse_date

from .models import ArxivPaper, ArxivAuthor, ArxivCategory, ArxivLink, SearchResult, SearchParams
from .utils import format_arxiv_id, build_search_query

logger = logging.getLogger(__name__)


class ArxivClient:
    """Client for interacting with the arXiv API."""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    USER_AGENT = "arxiv-mcp/0.1.0"
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """
        Initialize the arXiv client.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": self.USER_AGENT}
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def _parse_entry(self, entry: Dict[str, Any]) -> ArxivPaper:
        """
        Parse a single entry from the arXiv API response.
        
        Args:
            entry: Raw entry from feedparser
            
        Returns:
            Parsed ArxivPaper object
        """
        # Extract arXiv ID from the entry ID
        arxiv_id = entry.id.split('/')[-1]
        
        # Parse authors
        authors = []
        if 'authors' in entry:
            for author in entry.authors:
                authors.append(ArxivAuthor(name=author.name))
        elif 'author' in entry:
            authors.append(ArxivAuthor(name=entry.author))
        
        # Parse categories
        categories = []
        primary_category = None
        
        if 'tags' in entry:
            for tag in entry.tags:
                category = ArxivCategory(
                    term=tag.term,
                    scheme=tag.scheme,
                    label=getattr(tag, 'label', None)
                )
                categories.append(category)
                
                # The first category is usually the primary one
                if primary_category is None:
                    primary_category = category
        
        # If no primary category found, create a default one
        if primary_category is None:
            primary_category = ArxivCategory(term="unknown", scheme="http://arxiv.org/schemas/atom")
        
        # Parse links
        links = []
        pdf_url = None
        abs_url = None
        
        if 'links' in entry:
            for link in entry.links:
                arxiv_link = ArxivLink(
                    href=link.href,
                    rel=link.rel,
                    type=getattr(link, 'type', None),
                    title=getattr(link, 'title', None)
                )
                links.append(arxiv_link)
                
                # Extract specific URLs
                if link.rel == 'alternate':
                    abs_url = link.href
                elif getattr(link, 'title', '') == 'pdf':
                    pdf_url = link.href
        
        # Parse dates
        published = parse_date(entry.published)
        updated = parse_date(entry.updated)
        
        # Extract additional metadata
        comment = getattr(entry, 'arxiv_comment', None)
        journal_ref = getattr(entry, 'arxiv_journal_ref', None)
        doi = getattr(entry, 'arxiv_doi', None)
        
        return ArxivPaper(
            id=arxiv_id,
            title=entry.title,
            summary=entry.summary,
            authors=authors,
            published=published,
            updated=updated,
            categories=categories,
            primary_category=primary_category,
            links=links,
            pdf_url=pdf_url,
            abs_url=abs_url,
            comment=comment,
            journal_ref=journal_ref,
            doi=doi
        )
    
    async def search(self, params: SearchParams) -> SearchResult:
        """
        Search for papers on arXiv.
        
        Args:
            params: Search parameters
            
        Returns:
            SearchResult containing papers and metadata
        """
        query_params = {
            'search_query': params.query,
            'start': params.start,
            'max_results': params.max_results,
            'sortBy': params.sort_by,
            'sortOrder': params.sort_order
        }
        
        url = f"{self.BASE_URL}?{urlencode(query_params)}"
        
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            
            # Parse the Atom feed
            feed = feedparser.parse(response.text)
            
            if 'bozo_exception' in feed and feed.bozo:
                logger.warning(f"Feed parsing warning: {feed.bozo_exception}")
            
            # Extract metadata
            total_results = int(feed.feed.opensearch_totalresults)
            start_index = int(feed.feed.opensearch_startindex)
            items_per_page = int(feed.feed.opensearch_itemsperpage)
            
            # Parse papers
            papers = []
            for entry in feed.entries:
                try:
                    paper = self._parse_entry(entry)
                    papers.append(paper)
                except Exception as e:
                    logger.error(f"Error parsing entry {entry.get('id', 'unknown')}: {e}")
                    continue
            
            return SearchResult(
                papers=papers,
                total_results=total_results,
                start_index=start_index,
                items_per_page=items_per_page,
                query=params.query
            )
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during search: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise
    
    async def get_paper_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Get a specific paper by its arXiv ID.
        
        Args:
            arxiv_id: The arXiv identifier
            
        Returns:
            ArxivPaper if found, None otherwise
        """
        formatted_id = format_arxiv_id(arxiv_id)
        
        query_params = {
            'id_list': formatted_id,
            'max_results': 1
        }
        
        url = f"{self.BASE_URL}?{urlencode(query_params)}"
        
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            
            feed = feedparser.parse(response.text)
            
            if not feed.entries:
                return None
            
            return self._parse_entry(feed.entries[0])
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error getting paper {arxiv_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting paper {arxiv_id}: {e}")
            raise
    
    async def search_by_author(self, author_name: str, max_results: int = 10) -> SearchResult:
        """
        Search for papers by a specific author.
        
        Args:
            author_name: Name of the author to search for
            max_results: Maximum number of results
            
        Returns:
            SearchResult containing papers by the author
        """
        query = build_search_query(author=author_name)
        params = SearchParams(
            query=query,
            max_results=max_results,
            sort_by="submittedDate",
            sort_order="descending"
        )
        return await self.search(params)
    
    async def search_by_category(self, category: str, max_results: int = 10) -> SearchResult:
        """
        Search for papers in a specific category.
        
        Args:
            category: arXiv category (e.g., 'cs.AI', 'physics.gen-ph')
            max_results: Maximum number of results
            
        Returns:
            SearchResult containing papers in the category
        """
        query = build_search_query(category=category)
        params = SearchParams(
            query=query,
            max_results=max_results,
            sort_by="submittedDate",
            sort_order="descending"
        )
        return await self.search(params)
    
    async def download_pdf(self, arxiv_id: str) -> bytes:
        """
        Download the PDF content of a paper.
        
        Args:
            arxiv_id: The arXiv identifier
            
        Returns:
            PDF content as bytes
        """
        formatted_id = format_arxiv_id(arxiv_id)
        pdf_url = f"https://arxiv.org/pdf/{formatted_id}.pdf"
        
        try:
            response = await self._client.get(pdf_url)
            response.raise_for_status()
            return response.content
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error downloading PDF {arxiv_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading PDF {arxiv_id}: {e}")
            raise
