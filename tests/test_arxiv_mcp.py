"""Test suite for the arXiv MCP server."""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock

from src.arxiv_mcp.arxiv_client import ArxivClient
from src.arxiv_mcp.models import SearchParams, ArxivPaper, ArxivAuthor, ArxivCategory
from src.arxiv_mcp.utils import format_arxiv_id, build_search_query, format_paper_summary


class TestUtils:
    """Test utility functions."""
    
    def test_format_arxiv_id_new_format(self):
        """Test formatting new arXiv ID format."""
        assert format_arxiv_id("2301.00001") == "2301.00001"
        assert format_arxiv_id("2301.00001v1") == "2301.00001"
        assert format_arxiv_id("https://arxiv.org/abs/2301.00001v2") == "2301.00001"
    
    def test_format_arxiv_id_old_format(self):
        """Test formatting old arXiv ID format."""
        assert format_arxiv_id("cs/0601001") == "cs/0601001"
        assert format_arxiv_id("cs/0601001v1") == "cs/0601001"
    
    def test_format_arxiv_id_invalid(self):
        """Test invalid arXiv ID format."""
        with pytest.raises(ValueError):
            format_arxiv_id("invalid-id")
        with pytest.raises(ValueError):
            format_arxiv_id("123.456")
    
    def test_build_search_query(self):
        """Test search query building."""
        query = build_search_query(title="machine learning", author="Smith")
        assert "title:machine learning" in query or 'title:"machine learning"' in query
        assert "author:Smith" in query
        assert "AND" in query
    
    def test_build_search_query_empty(self):
        """Test search query with no parameters."""
        query = build_search_query()
        assert query == ""


class TestModels:
    """Test data models."""
    
    def test_search_params_validation(self):
        """Test SearchParams validation."""
        params = SearchParams(query="test", max_results=50)
        assert params.max_results == 50
        
        # Should fail validation due to max_results > 100
        with pytest.raises(Exception):  # Pydantic ValidationError
            SearchParams(query="test", max_results=150)
    
    def test_arxiv_paper_model(self):
        """Test ArxivPaper model creation."""
        from datetime import datetime
        
        paper = ArxivPaper(
            id="2301.00001",
            title="Test Paper",
            summary="Test summary",
            authors=[ArxivAuthor(name="Test Author")],
            published=datetime.now(),
            updated=datetime.now(),
            categories=[ArxivCategory(term="cs.AI", scheme="http://arxiv.org/schemas/atom")],
            primary_category=ArxivCategory(term="cs.AI", scheme="http://arxiv.org/schemas/atom"),
            links=[]
        )
        
        assert paper.id == "2301.00001"
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 1
        assert paper.authors[0].name == "Test Author"


@pytest.mark.asyncio
class TestArxivClient:
    """Test arXiv API client."""
    
    async def test_client_initialization(self):
        """Test client initialization."""
        client = ArxivClient(timeout=10, max_retries=2)
        assert client.timeout == 10
        assert client.max_retries == 2
        await client.close()
    
    async def test_client_context_manager(self):
        """Test client as context manager."""
        async with ArxivClient() as client:
            assert client is not None


# Integration test (requires internet connection)
@pytest.mark.integration
@pytest.mark.asyncio
class TestArxivIntegration:
    """Integration tests that require actual API calls."""
    
    async def test_search_integration(self):
        """Test actual search against arXiv API."""
        async with ArxivClient() as client:
            params = SearchParams(
                query="all:electron",
                max_results=1
            )
            result = await client.search(params)
            
            assert result.total_results > 0
            assert len(result.papers) == 1
            assert result.papers[0].id
            assert result.papers[0].title
    
    async def test_get_paper_by_id_integration(self):
        """Test getting a specific paper by ID."""
        async with ArxivClient() as client:
            # Use a well-known paper ID
            paper = await client.get_paper_by_id("1706.03762")  # Attention is All You Need
            
            assert paper is not None
            # arXiv IDs may include version suffixes (e.g., "1706.03762v7")
            assert paper.id.startswith("1706.03762")
            assert "attention" in paper.title.lower() or "transformer" in paper.title.lower()
            assert len(paper.authors) > 0


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v", "-m", "not integration"])
