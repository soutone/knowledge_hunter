# raw_scraper.py

import asyncio
import hashlib
import os
import re
import sys
import time
import json # Added for GitHub API parsing
import base64 # Added for decoding GitHub file content if needed
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urljoin, urlparse, unquote
from pathlib import Path
import httpx
from bs4 import BeautifulSoup
import traceback
import numpy as np

# --- Import Configuration ---
import config

# --- Import necessary functions from content_processor ---
try:
    from content_processor import (
        clean_html_content, # Still needed for web scraping part
        get_embedding,
        cosine_similarity,
    )
    CONTENT_PROCESSOR_AVAILABLE = True
except ImportError:
    print("[Scraper Error] Could not import functions from content_processor.py. Content filtering disabled.")
    CONTENT_PROCESSOR_AVAILABLE = False
    # Define dummy functions if import fails
    def clean_html_content(html: str) -> str: return ""
    async def get_embedding(text: str, model: Optional[str] = None) -> Optional[List[float]]: return None
    def cosine_similarity(v1: Optional[List[float]], v2: Optional[List[float]]) -> float: return 0.0

# --- Constants from Config ---
OUTPUT_RAW_DIR = config.OUTPUT_RAW_DIR
SAVE_RAW = config.SAVE_RAW

# Web Scraping Config
WEB_REQUEST_TIMEOUT = config.REQUEST_TIMEOUT
WEB_CONCURRENT_REQUEST_LIMIT = config.CONCURRENT_REQUEST_LIMIT

# GitHub Scraping Config
GITHUB_API_TOKEN = config.GITHUB_API_TOKEN
GITHUB_API_BASE_URL = config.GITHUB_API_BASE_URL
GITHUB_TARGET_EXTENSIONS = config.GITHUB_TARGET_EXTENSIONS
GITHUB_REQUEST_TIMEOUT = config.GITHUB_REQUEST_TIMEOUT
GITHUB_CONCURRENT_REQUEST_LIMIT = config.GITHUB_CONCURRENT_REQUEST_LIMIT

# Content Filtering Config
CONTENT_SEMANTIC_THRESHOLD = config.CONTENT_SEMANTIC_THRESHOLD

# --- User Agent ---
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
# --- GitHub API Headers ---
GITHUB_HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": HEADERS["User-Agent"], # Use the same user agent
}
if GITHUB_API_TOKEN:
    GITHUB_HEADERS["Authorization"] = f"Bearer {GITHUB_API_TOKEN}"
else:
    # This case was warned about in config.py, but add runtime check too
    print("[Scraper Warning] GitHub API Token not found. GitHub scraping will likely fail due to rate limits or permissions.")


# --- Helper Functions ---

def sanitize_filename(filename: str) -> str:
    """Remove or replace characters that are invalid in filenames."""
    # Remove potentially problematic URL schemes if present at start
    filename = re.sub(r"^[a-zA-Z]+://", "", filename)
    # Replace common invalid filename characters with underscores
    filename = re.sub(r'[\\/*?:"<>|]+', "_", filename)
    # Replace sequences of whitespace and underscores with a single underscore
    filename = re.sub(r"[\s_]+", "_", filename)
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    # Limit overall length to prevent issues on some filesystems
    max_len = 180
    if len(filename) > max_len:
        # Simple truncation if too long
        filename = filename[:max_len]
    return filename

def build_raw_filename(url: str, is_github: bool = False, github_path: Optional[str] = None) -> str:
    """Creates a filename for storing the raw content (HTML or Markdown) of a URL."""
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace(':', '_').replace('.', '_') if parsed_url.netloc else 'no_domain'

        if is_github and github_path:
            # For GitHub, use the repo path for a more descriptive name
            path_slug = sanitize_filename(github_path.strip('/'))
            # Keep extension from github_path
            _, ext = os.path.splitext(github_path)
            if not ext: ext = ".md" # Default if no extension found
            base_name = f"raw_{domain}_{path_slug}"

        else:
            # For web pages, use the URL path and query
            full_path = f"{parsed_url.path}?{parsed_url.query}" if parsed_url.query else parsed_url.path
            path_slug = sanitize_filename(full_path.strip('/')) if full_path.strip('/') else 'index'
             # Default extension for web pages
            ext = ".html"
            base_name = f"raw_{domain}_{path_slug}"

        # Add a short hash of the original URL/path to prevent collisions
        hash_input = github_path if is_github and github_path else url
        hash_part = hashlib.sha1(hash_input.encode('utf-8', 'ignore')).hexdigest()[:8]

        # Combine parts, ensuring total length is reasonable
        max_name_len = 200
        if len(base_name) > max_name_len:
            base_name = base_name[:max_name_len]

        # Ensure the extension is part of the final filename
        filename = f"{base_name}_{hash_part}{ext}"
        # Final sanitize just in case combination created issues
        return sanitize_filename(filename)

    except Exception as e:
        print(f"[FilenameError] Error building filename for {url} (GitHub Path: {github_path}): {e}. Using hash fallback.")
        traceback.print_exc()
        # Fallback filename if error occurs
        fallback_ext = ".md" if is_github else ".html"
        return f"raw_error_{hashlib.sha1(url.encode('utf-8', 'ignore')).hexdigest()[:16]}{fallback_ext}"


def save_raw_content(content: str, filepath: str):
    """Saves content (HTML or Markdown) to a file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Use utf-8 encoding, ignore errors for broader compatibility
        with open(filepath, "w", encoding="utf-8", errors="ignore") as f:
            f.write(content)
        # print(f"[SaveRaw] Saved raw content to {filepath}") # Reduce log noise
    except OSError as e:
        print(f"[SaveError] OS Error saving raw content to {filepath}: {e}")
    except Exception as e:
        print(f"[SaveError] Unexpected error saving {filepath}: {type(e).__name__} - {e}")
        traceback.print_exc()

# --- GitHub API Specific Functions ---

def is_github_repo_url(url: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Checks if a URL is a GitHub repository URL and extracts owner/repo."""
    try:
        parsed_url = urlparse(url)
        if parsed_url.netloc.lower() == 'github.com':
            path_parts = [part for part in parsed_url.path.split('/') if part]
            if len(path_parts) >= 2:
                # Basic check: /owner/repo
                owner = path_parts[0]
                repo = path_parts[1]
                # Avoid matching URLs like /login, /features, etc.
                if owner and repo and owner != 'topics':
                     # Further check to avoid matching /orgs/owner/projects etc.
                     if len(path_parts) == 2 or path_parts[2] in ['tree', 'blob', '']: # Allow root, tree, blob links
                        # print(f"[GitHubDetect] Detected GitHub repo: {owner}/{repo}")
                        return True, owner, repo
        return False, None, None
    except Exception as e:
        print(f"[GitHubDetect Error] Error parsing URL {url}: {e}")
        return False, None, None

async def fetch_raw_github_content(
    session: httpx.AsyncClient,
    download_url: str,
    semaphore: asyncio.Semaphore
) -> Optional[str]:
    """Fetches raw content from a GitHub download URL."""
    async with semaphore:
        try:
            # Use standard HEADERS for downloading raw content, no auth usually needed/beneficial here
            response = await session.get(download_url, headers=HEADERS, timeout=GITHUB_REQUEST_TIMEOUT)
            response.raise_for_status()
            # Decode assuming UTF-8, ignore errors
            return response.content.decode('utf-8', 'ignore')
        except httpx.TimeoutException:
            print(f"[GitHubFetch] Timeout fetching raw content {download_url} after {GITHUB_REQUEST_TIMEOUT}s")
        except httpx.RequestError as e:
            print(f"[GitHubFetch] Request error fetching raw content {download_url}: {type(e).__name__}")
        except httpx.HTTPStatusError as e:
            print(f"[GitHubFetch] HTTP error {e.response.status_code} fetching raw content {download_url}")
        except Exception as e:
            print(f"[GitHubFetch] Unknown error fetching raw content {download_url}: {type(e).__name__}")
            traceback.print_exc()
        return None

async def get_repo_contents(
    session: httpx.AsyncClient,
    owner: str,
    repo: str,
    path: str,
    semaphore: asyncio.Semaphore,
    fetch_tasks: List[Tuple[asyncio.Task, str, str]], # Changed name for clarity
    processed_files: Set[str] # Keep track of files for which fetch tasks are created
):
    """Recursively fetches repository contents via GitHub API and creates fetch tasks for Markdown files."""
    api_url = f"{GITHUB_API_BASE_URL}/repos/{owner}/{repo}/contents/{path}"
    # print(f"[GitHubAPI] Querying contents: {api_url}") # Reduce noise
    try:
        async with semaphore: # Use semaphore for the API listing call itself
            response = await session.get(api_url, headers=GITHUB_HEADERS, timeout=GITHUB_REQUEST_TIMEOUT)
            response.raise_for_status()
            contents = response.json()

        if not isinstance(contents, list):
            # Handle case where path points directly to a file
            if isinstance(contents, dict) and contents.get('type') == 'file':
                contents = [contents] # Treat as a list with one item
            else:
                print(f"[GitHubAPI Warning] Unexpected content format for {api_url}, expected list or file dict, got {type(contents)}. Skipping.")
                return

        tasks_to_await = [] # For recursive calls
        for item in contents:
            item_path = item.get('path')
            item_type = item.get('type')
            item_html_url = item.get('html_url') # URL to view file on GitHub

            if not item_path or not item_type or not item_html_url:
                print(f"[GitHubAPI Warning] Skipping item with missing path, type, or html_url in {api_url}")
                continue

            if item_type == 'file':
                _, extension = os.path.splitext(item_path)
                if extension.lower() in GITHUB_TARGET_EXTENSIONS:
                    download_url = item.get('download_url')
                    if download_url and item_html_url not in processed_files:
                        # print(f"[GitHubAPI] Found Markdown: {item_path} -> {download_url}") # Reduce noise
                        # Create task to fetch the raw content, store context
                        task = asyncio.create_task(fetch_raw_github_content(session, download_url, semaphore))
                        # Store task with its context (HTML URL and repo path) for later processing
                        fetch_tasks.append((task, item_html_url, item_path))
                        processed_files.add(item_html_url) # Mark as processed
                    elif not download_url:
                         print(f"[GitHubAPI Warning] File item '{item_path}' missing download_url. Skipping.")

            elif item_type == 'dir':
                # Recursively call for subdirectory
                # print(f"[GitHubAPI] Descending into directory: {item_path}") # Reduce noise
                # Use create_task to allow concurrent directory listings
                tasks_to_await.append(
                    asyncio.create_task(get_repo_contents(session, owner, repo, item_path, semaphore, fetch_tasks, processed_files))
                )

        if tasks_to_await:
            await asyncio.gather(*tasks_to_await) # Wait for recursive calls to complete

    except httpx.TimeoutException:
        print(f"[GitHubAPI] Timeout fetching contents for {api_url} after {GITHUB_REQUEST_TIMEOUT}s")
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 404:
            print(f"[GitHubAPI] Not Found (404) for path: {path} in {owner}/{repo}")
        elif status == 403: # Often rate limit or auth issue
            print(f"[GitHubAPI] Forbidden (403) for {api_url}. Check token permissions or rate limits.")
            # Could implement backoff here if needed
        elif status == 429: # Explicit rate limit
            print(f"[GitHubAPI] Rate Limit Exceeded (429) for {api_url}. Backing off...")
            # Basic backoff example (consider more sophisticated strategies)
            await asyncio.sleep(5)
        else:
            print(f"[GitHubAPI] HTTP error {status} fetching contents for {api_url}")
            try: print(f"  Response: {e.response.text[:200]}") # Log snippet of response
            except: pass
    except httpx.RequestError as e:
        print(f"[GitHubAPI] Request error fetching contents for {api_url}: {type(e).__name__}")
    except json.JSONDecodeError:
        print(f"[GitHubAPI] Error decoding JSON response for {api_url}")
    except Exception as e:
        print(f"[GitHubAPI] Unknown error processing contents for {api_url}: {type(e).__name__}")
        traceback.print_exc()

async def crawl_github_repo(
    start_url: str,
    owner: str,
    repo: str,
    topic_embedding: Optional[List[float]]
) -> Dict[str, Dict[str, Any]]:
    """Crawls a GitHub repository using the API to fetch Markdown files."""
    print(f"[CrawlGitHub] Starting API crawl for: {owner}/{repo}")
    print(f"[CrawlGitHub] Target Extensions: {GITHUB_TARGET_EXTENSIONS}")
    print(f"[CrawlGitHub] API Concurrent Request Limit: {GITHUB_CONCURRENT_REQUEST_LIMIT}")
    print(f"[CrawlGitHub] Save Raw Markdown: {SAVE_RAW}")
    semantic_status = f">= {CONTENT_SEMANTIC_THRESHOLD:.2f}" if CONTENT_SEMANTIC_THRESHOLD <= 1.0 and topic_embedding else "Disabled"
    print(f"[CrawlGitHub] Topic Embedding for Content Filter: {'Provided' if topic_embedding else 'Not Provided'} (Similarity {semantic_status})")
    print("-" * 40)

    scraped_data: Dict[str, Dict[str, Any]] = {}
    content_filtered_urls: Set[str] = set()
    urls_failed_fetch: Set[str] = set()
    processed_files_tracker: Set[str] = set() # Track files added to fetch_tasks

    fetch_semaphore = asyncio.Semaphore(GITHUB_CONCURRENT_REQUEST_LIMIT)
    fetch_tasks_with_context: List[Tuple[asyncio.Task, str, str]] = [] # Stores (task, html_url, repo_path)

    try:
        async with httpx.AsyncClient(verify=True) as session: # Re-use client
            # Start recursive fetching from the root directory
            await get_repo_contents(session, owner, repo, "", fetch_semaphore, fetch_tasks_with_context, processed_files_tracker)

            print(f"[CrawlGitHub] API traversal complete. Found {len(fetch_tasks_with_context)} potential Markdown files. Fetching content...")

            # --- Gather Raw Content ---
            raw_content_results: Dict[str, Tuple[Optional[str], str]] = {} # html_url -> (content, repo_path)
            tasks = [t[0] for t in fetch_tasks_with_context]
            if tasks:
                task_results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(task_results):
                    # Check if task completed successfully before accessing context
                    if i < len(fetch_tasks_with_context):
                        _, html_url, repo_path = fetch_tasks_with_context[i] # Get context back
                        if isinstance(result, Exception):
                            print(f"[CrawlGitHub] Error gathering fetch result for {html_url}: {type(result).__name__}")
                            urls_failed_fetch.add(html_url)
                        elif result is None:
                            # Fetch helper function already printed error
                            urls_failed_fetch.add(html_url)
                        else:
                            # Store successful result with context
                            raw_content_results[html_url] = (result, repo_path)
                    else:
                         print(f"[CrawlGitHub Warning] Result index {i} out of bounds for task context.")


            print(f"[CrawlGitHub] Content fetching complete. Results: {len(raw_content_results)} fetched, {len(urls_failed_fetch)} failed.")

            # --- Content Filtering and Saving ---
            print("[CrawlGitHub] Applying content filters...")
            filter_tasks = []
            filter_context: List[Tuple[str, str, str]] = [] # html_url, raw_content, repo_path

            for html_url, (raw_content, repo_path) in raw_content_results.items():
                if raw_content is not None: # Should always be true here, but check anyway
                    # Pass raw_content (Markdown) directly
                    task = asyncio.create_task(should_process_content(raw_content, html_url, topic_embedding, is_markdown=True))
                    filter_tasks.append(task)
                    filter_context.append((html_url, raw_content, repo_path))

            if filter_tasks:
                filter_results = await asyncio.gather(*filter_tasks, return_exceptions=True)
                for i, result in enumerate(filter_results):
                     # Check if task completed successfully before accessing context
                    if i < len(filter_context):
                        html_url, raw_content, repo_path = filter_context[i]
                        if isinstance(result, Exception):
                            print(f"[Content Check Error] Task for GitHub file {html_url} failed: {type(result).__name__} - {result}")
                            content_filtered_urls.add(html_url)
                        elif result is None: # None indicates filtered out
                            content_filtered_urls.add(html_url)
                        else: # Passed filter, result is the boolean True
                            # *** CHANGE HERE: Add type and use 'content' key ***
                            scraped_data[html_url] = {
                                "content": raw_content, # Use 'content' key
                                "type": "markdown",    # Add type identifier
                                "depth": 0,            # Depth less relevant, use 0
                                "path": repo_path      # Keep repo path
                            }

                            if SAVE_RAW:
                                filename = build_raw_filename(html_url, is_github=True, github_path=repo_path)
                                filepath = os.path.join(OUTPUT_RAW_DIR, filename)
                                save_raw_content(raw_content, filepath)
                    else:
                        print(f"[CrawlGitHub Warning] Result index {i} out of bounds for filter context.")


    except Exception as e:
        print(f"[CrawlGitHub] Unexpected error during GitHub crawl: {type(e).__name__}")
        traceback.print_exc()

    # --- Final Summary ---
    print("-" * 40)
    print(f"[CrawlGitHub] GitHub crawling finished.")
    total_files_considered = len(fetch_tasks_with_context)
    passed_filter_count = len(scraped_data)
    print(f"[CrawlGitHub Summary] Markdown Files Considered: {total_files_considered}. Files Passed Filter: {passed_filter_count}.")
    print(f"[CrawlGitHub] Files Failing Fetch: {len(urls_failed_fetch)}")
    print(f"[CrawlGitHub] Files Filtered by Content Check: {len(content_filtered_urls)}")
    print(f"[CrawlGitHub] Final Files Kept for Processing: {passed_filter_count}")
    if SAVE_RAW: print(f"[CrawlGitHub] Raw Markdown saved to: {OUTPUT_RAW_DIR}")
    print("-" * 40)
    return scraped_data


# --- Standard Web Scraping Functions ---

async def fetch_url(
    session: httpx.AsyncClient, url: str, semaphore: asyncio.Semaphore
) -> Optional[str]:
    """Fetches HTML content for a given URL asynchronously."""
    async with semaphore:
        try:
            response = await session.get(url, headers=HEADERS, timeout=WEB_REQUEST_TIMEOUT)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                try:
                    return response.text
                except UnicodeDecodeError:
                    print(f"[WebFetch] Unicode decode error for {url}, trying utf-8 ignore.")
                    return response.content.decode('utf-8', 'ignore')
            else:
                # Silently ignore non-html content for web crawl
                return None
        except httpx.TimeoutException:
            print(f"[WebFetch] Timeout fetching {url} after {WEB_REQUEST_TIMEOUT}s")
        except httpx.TooManyRedirects:
             print(f"[WebFetch] Too many redirects fetching {url}")
        except httpx.RequestError as e:
            # Log less common errors
            if "Connection refused" not in str(e) and "Connect call failed" not in str(e) and "Name or service not known" not in str(e):
                 print(f"[WebFetch] Request error fetching {url}: {type(e).__name__}")
        except httpx.HTTPStatusError as e:
             # Log non-404 errors, 429 might indicate general rate limit issues
             if e.response.status_code not in [404]:
                  print(f"[WebFetch] HTTP error fetching {url}: Status {e.response.status_code}")
        except Exception as e:
            print(f"[WebFetch] Unknown error fetching {url}: {type(e).__name__} - {e}")
            # traceback.print_exc() # Optional: more detailed logging
        return None

def extract_links(html_content: str, base_url: str) -> Set[str]:
    """Extracts and resolves absolute links from HTML content."""
    links = set()
    if not html_content: return links
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            # Ignore empty, anchor, mailto, javascript links
            if not href or href.startswith("#") or href.lower().startswith("mailto:") or href.lower().startswith("javascript:"):
                continue

            try:
                absolute_url = urljoin(base_url, href)
                parsed_url = urlparse(absolute_url)
                # Only keep HTTP/HTTPS URLs, remove fragments
                if parsed_url.scheme in ["http", "https"]:
                    cleaned_url = parsed_url._replace(fragment="").geturl()
                    links.add(cleaned_url)
            except ValueError: # Handle potential errors in urljoin/urlparse
                print(f"[LinkExtract Warning] Could not parse or resolve link '{href}' on page {base_url}")

    except Exception as e:
        # Catch potential BeautifulSoup errors
        print(f"[LinkExtract Error] Error parsing links from {base_url}: {type(e).__name__} - {e}")
    return links

async def should_queue_url(
    url: str,
    current_depth: int,
    max_depth: int,
    base_path: str,
    visited_urls: Set[str],
) -> Tuple[bool, str]:
    """Determines if a URL should be *queued for crawling* (Website context)."""
    reason_prefix = f"(d={current_depth})"
    # Already visited or fetch task created
    if url in visited_urls: return False, f"[{reason_prefix}] Visited/Queued"
    # Exceeds depth limit
    if current_depth > max_depth: return False, f"[{reason_prefix}] ❌ Exceeds max depth ({max_depth})"
    # Stays within the initially defined base path
    if not url.startswith(base_path): return False, f"[{reason_prefix}] ❌ Outside base path"
    # Only HTTP/HTTPS
    if not url.startswith(("http://", "https://")): return False, f"[{reason_prefix}] ❌ Not HTTP/HTTPS"

    # Basic file extension filtering (optional, can customize)
    try:
        parsed = urlparse(url)
        path = parsed.path.lower()
        excluded_extensions = {'.pdf', '.zip', '.exe', '.dmg', '.pkg', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.xml', '.rss'}
        if any(path.endswith(ext) for ext in excluded_extensions):
            return False, f"[{reason_prefix}] ❌ Excluded file type"
    except Exception:
        pass # Ignore parsing errors for this check

    return True, "" # Qualifies for queueing

async def should_process_content(
    content: str, # Can be HTML or Markdown
    url: str,
    topic_embedding: Optional[List[float]],
    is_markdown: bool = False # Flag to indicate content type
) -> Optional[bool]: # Return True if passed, None if failed/filtered
    """Determines if the *content* should be processed based on semantic similarity."""
    if not CONTENT_PROCESSOR_AVAILABLE:
        print(f"[Content Filter] Skipping for {url} - Content processor functions not available.")
        return True # Pass through if processor unavailable

    # --- MODIFIED: Handle Markdown vs HTML ---
    if is_markdown:
        # For Markdown, use the content directly
        text_for_embedding = content
        if not text_for_embedding:
             # print(f"[Content Filter] Skipping {url}: Empty Markdown content.")
             return None # Failed/Filtered
    else:
        # For HTML, clean it first
        cleaned_text = clean_html_content(content)
        if not cleaned_text:
            # print(f"[Content Filter] Skipping {url}: HTML cleaned to empty.")
            return None # Failed/Filtered
        text_for_embedding = cleaned_text
    # --- END MODIFICATION ---

    semantic_passed = False
    content_sem_score = 0.0
    semantic_enabled = CONTENT_SEMANTIC_THRESHOLD <= 1.0

    if topic_embedding is not None and semantic_enabled:
        try:
            content_embedding = await get_embedding(text_for_embedding) # Use appropriate text
            if content_embedding:
                # Ensure topic_embedding is not None before similarity check
                if topic_embedding:
                    content_sem_score = cosine_similarity(content_embedding, topic_embedding)
                    semantic_passed = content_sem_score >= CONTENT_SEMANTIC_THRESHOLD
                else: # Should not happen if topic_embedding is passed and not None initially, but safe check
                    print(f"[Content Filter] Warning: Topic embedding became None unexpectedly for {url}.")
                    semantic_passed = False # Cannot compare
            else:
                print(f"[Content Filter] Warning: Failed to generate content embedding for {url}. Semantic check failed.")
                semantic_passed = False
        except Exception as e:
             print(f"[Content Filter Error] Error during embedding/similarity for {url}: {type(e).__name__}. Semantic check failed.")
             traceback.print_exc()
             semantic_passed = False

    elif not semantic_enabled:
        semantic_passed = True # Pass if semantic filtering is disabled
        content_sem_score = -1.0 # Indicate semantic disabled
    else: # Semantic enabled, but no topic embedding provided
        semantic_passed = True # Pass if no topic embedding (cannot filter)
        content_sem_score = -2.0 # Indicate no topic embedding

    # Determine log prefix based on content type
    log_prefix = "[GitHub Content" if is_markdown else "[Web Content"

    if semantic_passed:
        sem_thresh_str = 'Disabled' if not semantic_enabled else f"{CONTENT_SEMANTIC_THRESHOLD:.2f}"
        if content_sem_score == -1.0: sem_score_str = "N/A (Sem Disabled)"
        elif content_sem_score == -2.0: sem_score_str = "N/A (No Topic Emb)"
        else: sem_score_str = f"{content_sem_score:.2f}"
        log_msg = f"✅ {log_prefix} Passed] (Score: {sem_score_str}, Threshold: {sem_thresh_str}) -> {url}"
        print(log_msg)
        return True # Indicate passed
    else:
        sem_thresh_str = f"{CONTENT_SEMANTIC_THRESHOLD:.2f}" # Should be enabled if we reach here
        sem_score_str = f"{content_sem_score:.2f}"
        log_msg = f"❌ {log_prefix} Rejected] (Score: {sem_score_str}, Threshold: {sem_thresh_str}) -> {url}"
        print(log_msg)
        return None # Indicate failed/filtered


async def crawl_website(
    start_url: str,
    max_depth: int,
    topic_embedding: Optional[List[float]] = None
) -> Dict[str, Dict[str, Any]]:
    """Crawls a standard website, applying semantic filters concurrently."""

    try:
        parsed_start_url = urlparse(start_url)
        if not parsed_start_url.scheme or not parsed_start_url.netloc:
            raise ValueError("Invalid start URL provided for website crawl.")
        # Define base path more strictly: scheme://netloc/path/ (if path exists)
        # Allows crawling within a specific documentation section
        base_path = urljoin(start_url, '.') # Resolves relative to the last part of the path

        # Optional: Define base domain for broader same-domain check if needed later
        # base_domain = parsed_start_url.netloc

    except Exception as e:
        print(f"[CrawlWebSetup] Error setting up website crawl for {start_url}: {type(e).__name__}. Cannot start crawl.")
        traceback.print_exc()
        return {}

    print(f"[CrawlWeb] Starting crawl from: {start_url}")
    print(f"[CrawlWeb] Max Depth: {max_depth}")
    # print(f"[CrawlWeb] Restricting crawl to base path: {base_path}") # Can be noisy
    semantic_status = f">= {CONTENT_SEMANTIC_THRESHOLD:.2f}" if CONTENT_SEMANTIC_THRESHOLD <= 1.0 and topic_embedding else "Disabled"
    print(f"[CrawlWeb] Topic Embedding for Content Filter: {'Provided' if topic_embedding else 'Not Provided'} (Similarity {semantic_status})")
    print(f"[CrawlWeb] Save Raw HTML: {SAVE_RAW}")
    print(f"[CrawlWeb] Concurrent Request Limit: {WEB_CONCURRENT_REQUEST_LIMIT}")
    print(f"[CrawlWeb] Request Timeout: {WEB_REQUEST_TIMEOUT}s")
    print("-" * 40)

    queue = asyncio.Queue()
    queue.put_nowait((start_url, 0)) # Item: (url, depth)
    # Tracks URLs added to queue or being fetched to prevent duplicates
    # We check should_queue_url *before* adding, so this acts as 'queued or being fetched'
    visited_urls_tracker: Set[str] = {start_url}
    # Stores data for pages that PASS content filter
    scraped_data: Dict[str, Dict[str, Any]] = {}
    fetch_semaphore = asyncio.Semaphore(WEB_CONCURRENT_REQUEST_LIMIT)
    # Tracks URLs that FAILED content filter (to avoid re-queuing from other pages)
    content_filtered_urls: Set[str] = set()
    # Tracks URLs that failed during fetch (timeout, error)
    urls_failed_fetch: Set[str] = set()

    active_tasks = 0

    async with httpx.AsyncClient(follow_redirects=True, timeout=WEB_REQUEST_TIMEOUT, verify=True) as session:
        while True:
            tasks_to_run = []
            # Launch new tasks if queue has items and concurrency allows
            while not queue.empty() and active_tasks < WEB_CONCURRENT_REQUEST_LIMIT:
                url, depth = queue.get_nowait()
                task = asyncio.create_task(process_single_url(
                    session, url, depth, max_depth, base_path, fetch_semaphore,
                    topic_embedding, visited_urls_tracker, content_filtered_urls, urls_failed_fetch, queue, scraped_data
                ))
                tasks_to_run.append(task)
                active_tasks += 1

            if not tasks_to_run and active_tasks == 0 and queue.empty():
                # print("[CrawlWeb] No running tasks and queue is empty. Exiting crawl loop.") # Less verbose
                break # Exit loop if nothing left to fetch or process

            if tasks_to_run:
                results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
                for i, result in enumerate(results):
                    active_tasks -= 1 # Decrement for each task completed
                    if isinstance(result, Exception):
                        # Extract original URL if possible from task context (needs adjustment if task context not passed)
                        # For now, just log the error
                        print(f"[CrawlWeb Task Error] Task failed: {type(result).__name__} - {result}")
                        # Optionally log traceback for debugging
                        # traceback.print_exception(type(result), result, result.__traceback__)

            elif active_tasks > 0:
                 # If no new tasks launched but some are still running, wait briefly
                 await asyncio.sleep(0.1)
            elif queue.empty(): # Double check if queue is empty after waiting
                # print("[CrawlWeb] Queue empty after waiting. Exiting loop.") # Less verbose
                break


    # --- Final Summary ---
    print("-" * 40)
    print(f"[CrawlWeb] Website crawling finished.")
    total_visited_or_queued = len(visited_urls_tracker)
    passed_filter_count = len(scraped_data)
    print(f"[CrawlWeb Summary] Pages Scanned (tasks created): {total_visited_or_queued}. Pages Passed Filter: {passed_filter_count}.")
    print(f"[CrawlWeb] URLs Failing Fetch: {len(urls_failed_fetch)}")
    print(f"[CrawlWeb] URLs Filtered by Content Check: {len(content_filtered_urls)}")
    print(f"[CrawlWeb] Final Pages Kept for Processing: {passed_filter_count}")
    if SAVE_RAW: print(f"[CrawlWeb] Raw HTML saved to: {OUTPUT_RAW_DIR}")
    print("-" * 40)
    return scraped_data

async def process_single_url(
    session: httpx.AsyncClient,
    url: str,
    depth: int,
    max_depth: int,
    base_path: str,
    semaphore: asyncio.Semaphore,
    topic_embedding: Optional[List[float]],
    visited_urls: Set[str], # Shared set for tracking
    content_filtered_urls: Set[str], # Shared set
    urls_failed_fetch: Set[str], # Shared set
    queue: asyncio.Queue, # Shared queue
    scraped_data: Dict[str, Dict[str, Any]] # Shared dict to store results
):
    """Fetches, filters, processes, and queues links for a single URL (Website crawl)."""
    try:
        # Fetch HTML content
        html_content = await fetch_url(session, url, semaphore)

        if html_content is None:
            urls_failed_fetch.add(url)
            return # Stop processing this URL if fetch failed

        # Check content relevance (semantic filter)
        # Note: is_markdown=False for website crawl
        process_flag = await should_process_content(html_content, url, topic_embedding, is_markdown=False)

        if process_flag is None: # None indicates filtered out
            content_filtered_urls.add(url)
            return # Stop processing this URL if filtered

        # Content passed filter - Add to results
        # *** Use 'content' key and add 'type': 'html' ***
        scraped_data[url] = {
            "content": html_content, # Use 'content' key
            "type": "html",         # Add type identifier
            "depth": depth,
            "path": urlparse(url).path # Add URL path for context
        }

        # Save raw HTML if enabled
        if SAVE_RAW:
            filename = build_raw_filename(url, is_github=False)
            filepath = os.path.join(OUTPUT_RAW_DIR, filename)
            save_raw_content(html_content, filepath)

        # Extract and queue new links if depth allows
        if depth < max_depth:
            new_links = extract_links(html_content, url)
            queued_count = 0
            for link in new_links:
                 # Check if link should be queued before adding to visited_urls
                 should_queue_flag, reason = await should_queue_url(link, depth + 1, max_depth, base_path, visited_urls)

                 if should_queue_flag:
                      # Add to visited *only when* adding to queue
                      if link not in visited_urls:
                           visited_urls.add(link)
                           queue.put_nowait((link, depth + 1))
                           queued_count += 1
                 # else:
                      # Optional: Log why a link was skipped (can be very verbose)
                      # if "Visited" not in reason: print(f"[Queue Skip] {reason} -> {link}")

            # if queued_count > 0: print(f"[Queue Add] Queued {queued_count} new links from {url} (d={depth})") # Reduce noise

    except Exception as e:
        print(f"[ProcessUrl Error] Unexpected error processing URL {url}: {type(e).__name__}")
        traceback.print_exc()
        urls_failed_fetch.add(url) # Mark as failed if any unexpected error occurs


# --- Main Entry Point ---

async def crawl_site(
    start_url: str,
    max_depth: int,
    topic_embedding: Optional[List[float]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Crawls a given start URL. Detects if it's a GitHub repository
    and uses the appropriate scraping strategy (API for GitHub, standard crawl for websites).
    """
    if SAVE_RAW:
        Path(OUTPUT_RAW_DIR).mkdir(parents=True, exist_ok=True)

    is_github, owner, repo = is_github_repo_url(start_url)

    if is_github and owner and repo:
        # Use GitHub API scraping logic
        if not GITHUB_API_TOKEN:
             print("[Crawl Error] GitHub repository detected, but GITHUB_API_TOKEN is missing in config/environment. Aborting GitHub crawl.")
             return {}
        return await crawl_github_repo(start_url, owner, repo, topic_embedding)
    else:
        # Use standard website scraping logic
        return await crawl_website(start_url, max_depth, topic_embedding)


# --- Standalone Execution Logic (for testing basic fetch) ---
if __name__ == "__main__":
    print("--- Raw Scraper (Standalone Single Page Fetch Mode) ---")
    # This mode now ONLY tests the standard web fetch, not the GitHub API fetch.
    print(f"--- Using config: Timeout={WEB_REQUEST_TIMEOUT}s, Output Dir={OUTPUT_RAW_DIR} ---")
    target_url = ""
    while not target_url:
        try:
            url_input = input("Enter the exact URL of the single WEB page to fetch: ").strip()
            if not url_input: print("URL cannot be empty."); continue
            parsed = urlparse(url_input)
            if parsed.scheme in ["http", "https"] and parsed.netloc:
                 # Prevent testing GitHub URLs in this simple mode
                 is_git, _, _ = is_github_repo_url(url_input)
                 if is_git:
                     print("This standalone mode is for testing standard web fetch. Use the main script for GitHub repos.")
                     continue
                 target_url = url_input
            else: print("Invalid URL. Please include http:// or https://.")
        except EOFError: print("\nOperation cancelled."); sys.exit(0)
        except KeyboardInterrupt: print("\nOperation cancelled."); sys.exit(0)

    async def fetch_and_save_single_page(url: str):
        Path(OUTPUT_RAW_DIR).mkdir(parents=True, exist_ok=True)
        print(f"[SingleWebFetch] Fetching {url}...")
        semaphore = asyncio.Semaphore(1) # Limit to 1 for single fetch test
        async with httpx.AsyncClient(follow_redirects=True, timeout=WEB_REQUEST_TIMEOUT) as session:
            # Use fetch_url which is part of the standard web scraping logic
            html_content = await fetch_url(session, url, semaphore)

        if html_content is not None:
            # Use build_raw_filename for web pages
            filename = build_raw_filename(url, is_github=False)
            filepath = os.path.join(OUTPUT_RAW_DIR, filename)
            save_raw_content(html_content, filepath) # Use updated save function
            print(f"[SingleWebFetch] Successfully fetched and saved {url}\n              Output: {filepath}")
        else:
            print(f"[SingleWebFetch] Failed to fetch or process {url}. Check URL and network.")

    start_time = time.perf_counter()
    try:
        if sys.platform == "win32":
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(fetch_and_save_single_page(target_url))
    except KeyboardInterrupt: print("\n[SingleWebFetch] Operation cancelled.")
    except Exception as e: print(f"\n[SingleWebFetch Error] An unexpected error occurred: {type(e).__name__} - {e}"); traceback.print_exc()
    finally: end_time = time.perf_counter(); print(f"[SingleWebFetch] Finished in {end_time - start_time:.2f} seconds."); print("-" * 60)