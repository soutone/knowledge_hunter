# config.py
import os
from typing import List # Added for type hinting

# Added dotenv loading for consistency across modules
from dotenv import load_dotenv
load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# --- ADDED: GitHub API Token ---
GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")

if not OPENAI_API_KEY:
    print("[Config Warning] OPENAI_API_KEY environment variable not found. LLM features will be disabled.")
# --- ADDED: Warning for GitHub Token ---
if not GITHUB_API_TOKEN:
    print("[Config Warning] GITHUB_API_TOKEN environment variable not found. GitHub repository scraping will be limited or fail.")

# --- Core Settings ---
MAX_DEPTH: int = 1                     # Maximum crawl depth for standard website scraping. (Not typically used for GitHub repo scraping)
SAVE_RAW: bool = False                 # Save raw HTML/Markdown content of pages that pass content filtering.
SKIP_COMPILATION: bool = False         # Skip the final step of compiling all .txt files.

# --- Model Selection ---
# Recommended models: gpt-4o-mini, gpt-4o, gpt-4-turbo
EXTRACTION_MODEL: str = "gpt-4o"       # LLM for detailed content extraction.
CONSOLIDATION_MODEL: str = "gpt-4o"    # LLM for consolidating extracted details.
RATING_MODEL: str = "gpt-4o-mini"      # LLM for quality rating (can be faster/cheaper).
EMBEDDING_MODEL: str = "text-embedding-3-small" # Model for generating text embeddings.

# --- LLM Processing Configuration ---
CHUNK_SIZE_TOKENS: int = 7000          # Max tokens per chunk sent to LLM for extraction/analysis.
LLM_CONCURRENCY_LIMIT: int = 20        # Max concurrent calls to OpenAI API.

# --- Content Filtering ---
CONTENT_SEMANTIC_THRESHOLD: float = 0.3 # Minimum semantic similarity score (cosine similarity) for content relevance.
                                        # Set > 1.0 (e.g., 1.1) to disable semantic filtering.

# --- Scraping Configuration ---
REQUEST_TIMEOUT: float = 20.0          # Timeout in seconds for fetching a single URL (Websites & GitHub API).
CONCURRENT_REQUEST_LIMIT: int = 10     # Max concurrent HTTP requests during standard web scraping.

# --- ADDED: GitHub Scraping Configuration ---
GITHUB_API_BASE_URL: str = "https://api.github.com" # Base URL for GitHub API calls
GITHUB_TARGET_EXTENSIONS: List[str] = [".md", ".mdx", ".markdown"] # File extensions to target in GitHub repos
GITHUB_REQUEST_TIMEOUT: float = 30.0   # Potentially longer timeout specifically for GitHub API calls if needed
GITHUB_CONCURRENT_REQUEST_LIMIT: int = 5 # Separate concurrency limit for GitHub API calls to respect API rate limits

# --- Output Directories & Files ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Use abspath for reliability
OUTPUT_DIR: str = os.path.join(BASE_DIR, 'output')              # Directory for processed text files (summaries).
OUTPUT_RAW_DIR: str = os.path.join(BASE_DIR, 'output_raw')      # Directory for raw HTML/Markdown files (if SAVE_RAW is True).
DEFAULT_COMPILED_OUTPUT_FILE: str = os.path.join(BASE_DIR, 'documentation.txt') # Default name for the final compiled file.
OUTPUT_QUALITY_DIR: str = os.path.join(BASE_DIR, 'output_quality') # Directory for quality assessment JSON files

# --- Tiktoken Configuration ---
TIKTOKEN_ENCODING: str = "cl100k_base" # Encoding used by many recent OpenAI models.

# --- User Confirmation ---
# Set to True to automatically use the first model option presented without asking (useful for automated runs).
AUTO_CONFIRM_MODELS: bool = False

print("[Config] Configuration loaded from config.py")