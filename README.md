# 📚 arXiv MCP Server

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-00d4aa?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K)](https://modelcontextprotocol.io/)
[![arXiv API](https://img.shields.io/badge/arXiv-API%20Integration-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://info.arxiv.org/help/api/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Code Quality](https://img.shields.io/badge/Code%20Quality-Black-000000?style=for-the-badge&logo=python&logoColor=white)](https://github.com/psf/black)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/tejas242/arxiv-mcp/ci.yml?branch=main&style=for-the-badge&logo=github&label=CI%2FCD)](https://github.com/tejas242/arxiv-mcp/actions)

</div>

> *Access the world's largest repository of academic papers through the Model Context Protocol*

A streamlined [Model Context Protocol](https://modelcontextprotocol.io/) server that connects AI assistants to arXiv's vast collection of academic papers. Search, analyze, and download research papers directly from your AI workflow.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone and setup
git clone https://github.com/tejas242/arxiv-mcp.git
cd arxiv-mcp
uv sync

# Test the server
uv run main.py
```

## 🛠️ Available Functions

<div align="center">

| Function | Status | Description | Parameters |
|----------|--------|-------------|------------|
| `search_papers` | ✅ **Working** | Search arXiv papers with flexible query syntax | `query`, `max_results`, `sort_by`, `sort_order` |
| `get_paper_details` | ✅ **Working** | Retrieve complete metadata for any arXiv paper | `arxiv_id` |
| `build_advanced_query` | ✅ **Working** | Construct complex search queries with multiple fields | `title_keywords`, `author_name`, `category`, `abstract_keywords` |
| `get_arxiv_categories` | ✅ **Working** | List all available arXiv subject categories | None |
| `search_by_author` | ⚠️ **Limited** | Find papers by specific author (use search_papers instead) | `author_name`, `max_results` |
| `search_by_category` | ⚠️ **Limited** | Browse papers by category (use search_papers instead) | `category`, `max_results` |
| `download_paper_pdf` | 🔧 **Needs Fix** | Download paper PDFs (redirect handling issue) | `arxiv_id`, `save_path` |

</div>

### Function Details

#### ✅ Fully Working Functions

**`search_papers`** - The primary search function
- Supports full arXiv query syntax
- Handles keywords, authors, categories, titles
- Configurable sorting and pagination
- Returns formatted results with abstracts and links

**`get_paper_details`** - Detailed paper information
- Complete metadata extraction
- Author information with affiliations
- Category classifications and links
- Publication dates and updates

**`build_advanced_query`** - Query construction helper
- Combines multiple search criteria
- Supports title, author, category, and abstract searches
- Returns properly formatted query strings

**`get_arxiv_categories`** - Category reference
- Complete list of arXiv subject categories
- Descriptions for each category
- Helpful for constructing targeted searches

#### ⚠️ Limited Functions (Workarounds Available)

**`search_by_author`** - Use `search_papers('au:"Author Name"')` instead
**`search_by_category`** - Use `search_papers('cat:category_code')` instead

#### 🔧 Functions Needing Fixes

**`download_paper_pdf`** - HTTP redirect handling needs improvement
- Currently fails due to HTTPS/HTTP redirect issues
- PDFs can be accessed directly via the links provided in search results

## ⚙️ Configuration

### Claude Desktop Setup

<details>
<summary><strong>Configuration Instructions</strong></summary>

Add to your Claude Desktop config file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "arxiv-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/arxiv-mcp",
        "run",
        "main.py"
      ]
    }
  }
}
```

</details>

### VS Code MCP Extension

<details>
<summary><strong>VS Code Configuration</strong></summary>

```json
{
  "mcp": {
    "servers": {
      "arxiv-mcp": {
        "command": "uv",
        "args": ["--directory", "/path/to/arxiv-mcp", "run", "main.py"]
      }
    }
  }
}
```

</details>

## 💡 Usage Examples

### Core Search Operations

```python
# Search for papers about transformers
search_papers("transformer architecture")

# Advanced query with specific fields
search_papers('ti:"attention mechanism" AND cat:cs.LG')

# Author-specific search (recommended approach)
search_papers('au:"Geoffrey Hinton"')

# Category browsing (recommended approach)
search_papers('cat:cs.AI')
```

### Research Workflow

```python
# 1. Find the famous "Attention" paper
search_papers('ti:"Attention Is All You Need"')
get_paper_details("1706.03762")

# 2. Explore related work
search_papers("transformer neural networks")

# 3. Build complex queries
query = build_advanced_query(
    title_keywords="few-shot learning",
    author_name="Tom Brown",
    category="cs.LG"
)
search_papers(query)
```

## 📊 arXiv Categories Reference

<details>
<summary><strong>Popular Categories</strong></summary>

| Code | Description | Example Topics |
|------|-------------|----------------|
| `cs.AI` | Artificial Intelligence | Machine learning, neural networks, AI theory |
| `cs.LG` | Machine Learning | Deep learning, reinforcement learning, statistical learning |
| `cs.CV` | Computer Vision | Image processing, object detection, visual recognition |
| `cs.CL` | Computation and Language | NLP, language models, text processing |
| `cs.CR` | Cryptography and Security | Security protocols, encryption, privacy |
| `stat.ML` | Machine Learning (Statistics) | Statistical learning theory, Bayesian methods |
| `physics.gen-ph` | General Physics | Theoretical physics, quantum mechanics |
| `math.NA` | Numerical Analysis | Computational mathematics, algorithms |
| `q-bio.NC` | Quantitative Biology | Neuroscience, computational biology |

</details>

Use `get_arxiv_categories()` for the complete list of available categories.

## 🧪 Testing Results

Based on comprehensive testing of all functions:

<div align="center">

![Working Functions](https://img.shields.io/badge/✅%20Working-4%20Functions-28a745?style=for-the-badge&logoColor=white)
![Limited Functions](https://img.shields.io/badge/⚠️%20Limited-2%20Functions-ffc107?style=for-the-badge&logoColor=black)
![Needs Fix](https://img.shields.io/badge/🔧%20Needs%20Fix-1%20Function-dc3545?style=for-the-badge&logoColor=white)

</div>

### ✅ Reliable Functions
- **Paper search with keywords, authors, categories**: 100% success rate
- **Paper detail retrieval**: Complete metadata extraction working
- **Query construction**: All syntax combinations supported
- **Category listing**: All arXiv categories accessible

### ⚠️ Alternative Approaches Recommended
- **Author search**: Use `search_papers('au:"Author Name"')` instead of `search_by_author()`
- **Category browsing**: Use `search_papers('cat:category')` instead of `search_by_category()`

### 🔧 Known Issues
- **PDF downloads**: Redirect handling needs improvement (PDFs accessible via direct links)

## 🔧 Development

### Project Structure
```
arxiv-mcp/
├── src/arxiv_mcp/          # Main package
│   ├── server.py           # MCP server implementation
│   ├── arxiv_client.py     # arXiv API wrapper
│   ├── models.py           # Pydantic data models
│   └── utils.py            # Helper functions
├── tests/                  # Test suite
├── main.py                 # Entry point
└── pyproject.toml         # Project config
```

### Running Tests
```bash
uv run pytest tests/ -v
```

### Debug Mode
```bash
# Enable detailed logging
PYTHONPATH=src uv run python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from arxiv_mcp.server import main
main()
"
```

## ⚠️ Troubleshooting

<details>
<summary><strong>Common Issues & Solutions</strong></summary>

### Server Not Detected
- ✅ Verify absolute paths in MCP config
- ✅ Test server runs: `uv run main.py`
- ✅ Restart Claude Desktop after config changes

### Search Issues
- ✅ Use arXiv query syntax (see examples above)
- ✅ Check category names: `get_arxiv_categories()`
- ✅ Try broader search terms
- ✅ Use `search_papers()` instead of specific search functions

### PDF Download Failures
- ✅ Access PDFs via links in search results
- ✅ Check internet connection
- ✅ Verify arXiv ID format (e.g., "1706.03762")

</details>

## 🙏 Acknowledgments

- **[arXiv](https://arxiv.org/)**
- **[Model Context Protocol](https://modelcontextprotocol.io/)**  

---

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tejas242/arxiv-mcp)
[![Issues](https://img.shields.io/badge/Report-Issues-red?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tejas242/arxiv-mcp/issues)
[![Contribute](https://img.shields.io/badge/Contribute-Welcome-brightgreen?style=for-the-badge&logo=git&logoColor=white)](https://github.com/tejas242/arxiv-mcp/pulls)

<br><br>

**Made with ⚡ by screenager**

</div>