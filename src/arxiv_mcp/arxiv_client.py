"""arXiv API client for retrieving papers and metadata."""

import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urlencode

import httpx
import feedparser
from dateutil.parser import parse as parse_date
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .models import ArxivPaper, ArxivAuthor, ArxivCategory, ArxivLink, SearchResult, SearchParams
from .utils import format_arxiv_id, build_search_query
from .config import get_settings
from .exceptions import (
    ArxivAPIError,
    InvalidArxivIdError,
    NetworkError,
    PaperNotFoundError,
    RateLimitError,
    SearchError,
    DownloadError,
    ValidationError,
)
from .logging_config import get_logger, log_api_request, log_api_response, log_error

logger = get_logger(__name__)


class ArxivClient:
    """Client for interacting with the arXiv API."""
    
    def __init__(self, timeout: Optional[int] = None, max_retries: Optional[int] = None):
        """
        Initialize the arXiv client.
        
        Args:
            timeout: Request timeout in seconds (overrides config)
            max_retries: Maximum number of retry attempts (overrides config)
        """
        try:
            settings = get_settings()
            self.config = settings.arxiv_api
            self.security_config = settings.security
        except Exception as e:
            logger.warning(f"Failed to load config, using defaults: {e}")
            # Fallback defaults
            class DefaultConfig:
                base_url = "http://export.arxiv.org/api/query"
                user_agent = "arxiv-mcp/0.1.0"
                timeout = 30
                max_retries = 3
                retry_delay = 1.0
                max_retry_delay = 60.0
                max_results_limit = 100
            
            class DefaultSecurityConfig:
                max_download_size = 100_000_000
                allowed_domains = ["arxiv.org", "export.arxiv.org"]
                sanitize_inputs = True
                validate_arxiv_ids = True
            
            self.config = DefaultConfig()
            self.security_config = DefaultSecurityConfig()
        
        # Override with provided values
        self.timeout = timeout or self.config.timeout
        self.max_retries = max_retries or self.config.max_retries
        
        # Setup HTTP client with enhanced configuration
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={"User-Agent": self.config.user_agent},
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=30
            ),
            follow_redirects=True,
        )
        
        # Rate limiting
        self._last_request_time = 0.0
        self._request_count = 0
        self._rate_limit_window_start = time.time()
    
    async def close(self):
        """Close the HTTP client."""
        if hasattr(self, '_client'):
            await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Reset window if needed
        if current_time - self._rate_limit_window_start >= self.config.rate_limit_window:
            self._rate_limit_window_start = current_time
            self._request_count = 0
        
        # Check limit
        if self._request_count >= self.config.rate_limit_requests:
            wait_time = self.config.rate_limit_window - (current_time - self._rate_limit_window_start)
            if wait_time > 0:
                raise RateLimitError(retry_after=int(wait_time))
        
        self._request_count += 1
    
    def _create_retry_decorator(self):
        """Create a retry decorator with configured settings."""
        return retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=self.config.retry_delay,
                max=self.config.max_retry_delay
            ),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
    
    async def _make_request(self, url: str) -> httpx.Response:
        """
        Make an HTTP request with retry logic and error handling.
        
        Args:
            url: URL to request
            
        Returns:
            HTTP response
            
        Raises:
            NetworkError: For network-related errors
            ArxivAPIError: For API-related errors
            RateLimitError: For rate limit violations
        """
        self._check_rate_limit()
        
        start_time = time.time()
        log_api_request(url)
        
        @self._create_retry_decorator()
        async def _request():
            try:
                response = await self._client.get(url)
                response_time = time.time() - start_time
                log_api_response(url, response.status_code, response_time)
                
                if response.status_code == 429:
                    retry_after = response.headers.get('Retry-After')
                    raise RateLimitError(retry_after=int(retry_after) if retry_after else None)
                elif response.status_code >= 500:
                    raise ArxivAPIError(
                        f"Server error: {response.status_code}",
                        status_code=response.status_code,
                        response_text=response.text
                    )
                elif response.status_code >= 400:
                    raise ArxivAPIError(
                        f"Client error: {response.status_code}",
                        status_code=response.status_code,
                        response_text=response.text
                    )
                
                response.raise_for_status()
                return response
                
            except httpx.HTTPError as e:
                log_error(e, context={'url': url})
                raise NetworkError(f"HTTP error: {str(e)}", original_error=e)
            except Exception as e:
                log_error(e, context={'url': url})
                raise ArxivAPIError(f"Unexpected error: {str(e)}")
        
        return await _request()
    
    def _validate_domain(self, url: str) -> None:
        """
        Validate that the URL domain is allowed.
        
        Args:
            url: URL to validate
            
        Raises:
            ValidationError: If domain is not allowed
        """
        from urllib.parse import urlparse
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
        
        if not any(allowed in domain for allowed in self.security_config.allowed_domains):
            raise ValidationError(f"Domain not allowed: {domain}")
    
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
        try:
            published = parse_date(entry.published)
            updated = parse_date(entry.updated)
        except Exception as e:
            logger.warning(f"Error parsing dates for {arxiv_id}: {e}")
            published = datetime.now()
            updated = datetime.now()
        
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
            
        Raises:
            ValidationError: For invalid search parameters
            SearchError: For search-related errors
            NetworkError: For network-related errors
        """
        # Validate parameters
        if params.max_results > self.config.max_results_limit:
            raise ValidationError(
                f"max_results cannot exceed {self.config.max_results_limit}",
                field="max_results",
                value=params.max_results
            )
        
        if self.security_config.sanitize_inputs:
            # Basic input sanitization
            sanitized_query = params.query.replace('\n', ' ').replace('\r', ' ')
            if sanitized_query != params.query:
                logger.warning("Query was sanitized")
                params.query = sanitized_query
        
        query_params = {
            'search_query': params.query,
            'start': params.start,
            'max_results': params.max_results,
            'sortBy': params.sort_by,
            'sortOrder': params.sort_order
        }
        
        url = f"{self.config.base_url}?{urlencode(query_params)}"
        self._validate_domain(url)
        
        try:
            response = await self._make_request(url)
            
            # Parse the Atom feed
            feed = feedparser.parse(response.text)
            
            if 'bozo_exception' in feed and feed.bozo:
                logger.warning(f"Feed parsing warning: {feed.bozo_exception}")
            
            # Check for feed-level errors
            if hasattr(feed, 'status') and feed.status >= 400:
                raise SearchError(f"Feed error: {feed.status}")
            
            # Extract metadata
            total_results = int(feed.feed.get('opensearch_totalresults', 0))
            start_index = int(feed.feed.get('opensearch_startindex', 0))
            items_per_page = int(feed.feed.get('opensearch_itemsperpage', 0))
            
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
            
        except (NetworkError, ValidationError, SearchError):
            raise
        except Exception as e:
            log_error(e, context={'query': params.query, 'url': url})
            raise SearchError(f"Unexpected search error: {str(e)}", query=params.query)
    
    async def get_paper_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Get a specific paper by its arXiv ID.
        
        Args:
            arxiv_id: The arXiv identifier
            
        Returns:
            ArxivPaper if found, None otherwise
            
        Raises:
            InvalidArxivIdError: For invalid arXiv ID format
            PaperNotFoundError: If paper is not found
            NetworkError: For network-related errors
        """
        try:
            if self.security_config.validate_arxiv_ids:
                formatted_id = format_arxiv_id(arxiv_id)
            else:
                formatted_id = arxiv_id
        except ValueError as e:
            raise InvalidArxivIdError(arxiv_id)
        
        query_params = {
            'id_list': formatted_id,
            'max_results': 1
        }
        
        url = f"{self.config.base_url}?{urlencode(query_params)}"
        self._validate_domain(url)
        
        try:
            response = await self._make_request(url)
            
            feed = feedparser.parse(response.text)
            
            if not feed.entries:
                raise PaperNotFoundError(arxiv_id)
            
            return self._parse_entry(feed.entries[0])
            
        except (NetworkError, PaperNotFoundError, InvalidArxivIdError):
            raise
        except Exception as e:
            log_error(e, context={'arxiv_id': arxiv_id, 'url': url})
            raise ArxivAPIError(f"Unexpected error getting paper {arxiv_id}: {str(e)}")
    
    async def search_by_author(self, author_name: str, max_results: int = 10) -> SearchResult:
        """
        Search for papers by a specific author.
        
        Args:
            author_name: Name of the author to search for
            max_results: Maximum number of results
            
        Returns:
            SearchResult containing papers by the author
        """
        if self.security_config.sanitize_inputs:
            author_name = author_name.strip()
        
        query = build_search_query(author=author_name)
        params = SearchParams(
            query=query,
            max_results=min(max_results, self.config.max_results_limit),
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
        if self.security_config.sanitize_inputs:
            category = category.strip()
        
        query = build_search_query(category=category)
        params = SearchParams(
            query=query,
            max_results=min(max_results, self.config.max_results_limit),
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
            
        Raises:
            InvalidArxivIdError: For invalid arXiv ID format
            DownloadError: For download-related errors
            NetworkError: For network-related errors
        """
        try:
            if self.security_config.validate_arxiv_ids:
                formatted_id = format_arxiv_id(arxiv_id)
            else:
                formatted_id = arxiv_id
        except ValueError:
            raise InvalidArxivIdError(arxiv_id)
        
        pdf_url = f"https://arxiv.org/pdf/{formatted_id}.pdf"
        self._validate_domain(pdf_url)
        
        try:
            response = await self._make_request(pdf_url)
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.security_config.max_download_size:
                raise DownloadError(
                    f"PDF too large: {content_length} bytes",
                    arxiv_id=arxiv_id,
                    url=pdf_url
                )
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'application/pdf' not in content_type:
                logger.warning(f"Unexpected content type for PDF: {content_type}")
            
            return response.content
            
        except (NetworkError, DownloadError, InvalidArxivIdError):
            raise
        except Exception as e:
            log_error(e, context={'arxiv_id': arxiv_id, 'url': pdf_url})
            raise DownloadError(f"Unexpected error downloading PDF {arxiv_id}: {str(e)}", arxiv_id=arxiv_id, url=pdf_url)
