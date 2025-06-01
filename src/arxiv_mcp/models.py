"""Data models for arXiv papers and search results."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, HttpUrl


class ArxivAuthor(BaseModel):
    """Represents an author of an arXiv paper."""
    name: str
    affiliation: Optional[str] = None


class ArxivCategory(BaseModel):
    """Represents an arXiv subject category."""
    term: str
    scheme: str
    label: Optional[str] = None


class ArxivLink(BaseModel):
    """Represents a link to paper resources."""
    href: str
    rel: str
    type: Optional[str] = None
    title: Optional[str] = None


class ArxivPaper(BaseModel):
    """Represents a complete arXiv paper with all metadata."""
    id: str = Field(description="arXiv identifier (e.g., 2301.00001)")
    title: str
    summary: str
    authors: List[ArxivAuthor]
    published: datetime
    updated: datetime
    categories: List[ArxivCategory]
    primary_category: ArxivCategory
    links: List[ArxivLink]
    pdf_url: Optional[str] = None
    abs_url: Optional[str] = None
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None


class SearchResult(BaseModel):
    """Represents search results from arXiv."""
    papers: List[ArxivPaper]
    total_results: int
    start_index: int
    items_per_page: int
    query: str


class SearchParams(BaseModel):
    """Parameters for searching arXiv papers."""
    query: str = Field(description="Search query using arXiv query syntax")
    max_results: int = Field(default=10, le=100, description="Maximum number of results to return")
    start: int = Field(default=0, description="Starting index for pagination")
    sort_by: str = Field(default="relevance", description="Sort order: relevance, lastUpdatedDate, submittedDate")
    sort_order: str = Field(default="descending", description="Sort direction: ascending, descending")
