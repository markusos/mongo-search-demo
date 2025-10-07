# GitHub Copilot Instructions for mongo-search-demo

## Project-Specific Guidelines

### Python Environment and Package Management

This project uses **uv** for Python package management and dependency resolution.

#### Running Python Commands

Always use `uv run` prefix when executing Python commands:

```bash
# Run pytest
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_text_processor.py -v

# Run Python scripts
uv run python scripts/ingest.py

# Run Python modules
uv run python -m pytest
```

**DO NOT USE:**
- `python` or `python3` directly
- `pip install` directly
- Virtual environment activation commands

#### Why uv?

- `uv` automatically manages the Python environment defined in `pyproject.toml`
- Ensures consistent dependencies across all developers
- Faster than traditional pip/venv workflows
- No need to manually activate/deactivate virtual environments

### Project Structure

- **src/**: Main source code modules
- **scripts/**: Executable scripts (ingest, search, benchmark)
- **tests/**: Test suite using pytest
- **config/**: Configuration files (config.yaml)
- **data/**: Wikipedia XML data files
- **embedding_cache/**: Cached embeddings for performance

### Code Standards

#### Enums

When defining configuration options with a fixed set of values, use Enums:

```python
from enum import Enum

class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    SEMANTIC = "semantic"
    FIXED = "fixed"
    HYBRID = "hybrid"
```

- Define enums in `config_loader.py` when they're used in configuration
- Use enum values in code instead of string literals
- Enums provide type safety and autocomplete support

### Testing

Run tests with:
```bash
uv run pytest tests/ -v
```

All changes should include corresponding test updates.
