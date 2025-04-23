# config.py
import os

# --- Core Settings ---
MAX_DEPTH: int = 1                     # Maximum crawl depth for the scraper.
SAVE_RAW: bool = False                 # Save raw HTML of pages that pass content filtering to OUTPUT_RAW_DIR.
SKIP_COMPILATION: bool = False         # Skip the final step of compiling all .txt files.

# --- Model Selection ---
EXTRACTION_MODEL: str = "gpt-4o-mini"             # LLM model for detailed content extraction.
CONSOLIDATION_MODEL: str = "gpt-4o-mini"          # LLM model for consolidating extracted details.
RATING_MODEL: str = "gpt-4o-mini"            # LLM model for quality rating (can be same as consolidation).
# KEYWORD_MODEL removed as keyword filtering is disabled
EMBEDDING_MODEL: str = "text-embedding-3-small" # Model for generating text embeddings (for semantic search).

# --- LLM Processing Configuration ---
CHUNK_SIZE_TOKENS: int = 7000          # Max tokens per chunk sent to LLM for extraction/analysis.
LLM_CONCURRENCY_LIMIT: int = 20        # Max concurrent calls to OpenAI API to manage rate limits/costs.

# --- Content Filtering ---
# Keyword filtering is removed. Only semantic filtering remains.
CONTENT_SEMANTIC_THRESHOLD: float = 0.3 # Minimum semantic similarity score (cosine similarity) for content relevance.
# Set > 1.0 (e.g., 1.1) to disable semantic filtering.

# --- Scraping Configuration ---
REQUEST_TIMEOUT: float = 20.0          # Timeout in seconds for fetching a single URL.
CONCURRENT_REQUEST_LIMIT: int = 10     # Max concurrent requests during scraping.

# --- Output Directories & Files ---
# Assumes 'output', 'output_raw' directories are relative to the script's location
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR: str = os.path.join(BASE_DIR, 'output')              # Directory for processed text files (summaries).
OUTPUT_RAW_DIR: str = os.path.join(BASE_DIR, 'output_raw')      # Directory for raw HTML files (if SAVE_RAW is True).
DEFAULT_COMPILED_OUTPUT_FILE: str = os.path.join(BASE_DIR, 'documentation.txt') # Default name for the final compiled file.

# --- Caching ---
# Filenames for caching mechanisms used in other scripts (optional, managed within those scripts)
# EMBEDDING_CACHE_FILE = "embedding_cache.json"
# FILENAME_CACHE_FILE = "filename_cache.json"

# --- Tiktoken Configuration ---
TIKTOKEN_ENCODING: str = "cl100k_base" # Encoding used by many recent OpenAI models.

print("[Config] Configuration loaded from config.py")