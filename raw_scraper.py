# raw_scraper.py

import asyncio
import hashlib
import os
import re
import sys # Keep sys for standalone execution exit
import time # Keep time for standalone execution timing
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urljoin, urlparse
from pathlib import Path # Keep Path for standalone execution directory creation
import httpx
from bs4 import BeautifulSoup # Keep BeautifulSoup for link extraction
import traceback
import numpy as np # Keep for cosine_similarity

# --- Import Configuration ---
import config # Import hardcoded config values

# --- Import necessary functions from content_processor ---
try:
    # Assuming content_processor is in the same directory or Python path
    from content_processor import (
        clean_html_content,
        get_embedding,
        cosine_similarity, # Moved here or keep in content_processor and import
        # tokenizer related imports might not be needed here anymore
    )
    CONTENT_PROCESSOR_AVAILABLE = True
except ImportError:
    print("[Scraper Error] Could not import functions from content_processor.py. Content filtering disabled.")
    CONTENT_PROCESSOR_AVAILABLE = False
    # Define dummy functions if import fails
    def clean_html_content(html): return ""
    async def get_embedding(text, model=None): return None
    def cosine_similarity(v1, v2): return 0.0

# --- Constants from Config ---
OUTPUT_RAW_DIR = config.OUTPUT_RAW_DIR
REQUEST_TIMEOUT = config.REQUEST_TIMEOUT
CONCURRENT_REQUEST_LIMIT = config.CONCURRENT_REQUEST_LIMIT
# CONTENT_KEYWORD_THRESHOLD removed
CONTENT_SEMANTIC_THRESHOLD = config.CONTENT_SEMANTIC_THRESHOLD
SAVE_RAW = config.SAVE_RAW

# --- User Agent ---
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# --- Helper Functions ---

def sanitize_filename(filename: str) -> str:
    """Remove or replace characters that are invalid in filenames."""
    filename = re.sub(r"^[a-zA-Z]+://", "", filename) # Remove protocol
    filename = re.sub(r'[\\/*?:"<>|]+', "_", filename) # Replace invalid chars
    filename = re.sub(r"__+", "_", filename) # Consolidate underscores
    max_len = 180 # Limit length
    if len(filename) > max_len:
        try:
            # Try to create a meaningful shortened name with hash
            if '://' not in filename: effective_url = "http://" + filename
            else: effective_url = filename
            parsed_url = urlparse(effective_url)
            domain = parsed_url.netloc.replace(':', '_') if parsed_url.netloc else 'no_domain'
            path_part = parsed_url.path + parsed_url.query
            hashed_path = hashlib.sha1(path_part.encode('utf-8', 'ignore')).hexdigest()[:8]
            filename = f"{domain}_{hashed_path}"[:max_len] # Ensure final length <= max_len
        except Exception:
            # Fallback to simpler hash if parsing fails
            safe_encoded = filename.encode('utf-8', 'ignore')
            filename = filename[:max_len-9] + "_" + hashlib.sha1(safe_encoded).hexdigest()[:8]
    return filename.strip('_')

def build_raw_filename(url: str) -> str:
    """Creates a filename for storing the raw HTML of a URL."""
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace(':', '_') if parsed_url.netloc else 'no_domain'
        # Use the path and query for the main part of the filename
        full_path = f"{parsed_url.path}?{parsed_url.query}" if parsed_url.query else parsed_url.path
        # Sanitize path/query to create a slug, or use 'index' if path is root
        path_slug = sanitize_filename(full_path.strip('/')) if full_path.strip('/') else 'index'
        hash_part = hashlib.sha1(full_path.encode('utf-8', 'ignore')).hexdigest()[:8] # Short hash
        # Combine parts, ensuring total length is reasonable
        base_name = f"raw_{domain}_{path_slug}"
        max_name_len = 200
        if len(base_name) > max_name_len: base_name = base_name[:max_name_len]
        filename = f"{base_name}_{hash_part}.html"
        # Final sanitize just in case combination created issues
        return sanitize_filename(filename)
    except Exception as e:
        print(f"[FilenameError] Error building filename for {url}: {e}. Using hash fallback.")
        traceback.print_exc()
        # Fallback filename if error occurs
        return f"raw_error_{hashlib.sha1(url.encode('utf-8', 'ignore')).hexdigest()[:16]}.html"

# Cosine similarity function might be imported from content_processor now or kept here if needed standalone

async def fetch_url(
    session: httpx.AsyncClient, url: str, semaphore: asyncio.Semaphore
) -> Optional[str]:
    """Fetches HTML content for a given URL asynchronously, respecting semaphore and timeout from config."""
    async with semaphore:
        try:
            # print(f"[Fetch] Requesting: {url}") # Reduce noise
            response = await session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                try:
                    return response.text
                except UnicodeDecodeError:
                    print(f"[Fetch] Unicode decode error for {url}, trying utf-8 ignore.")
                    return response.content.decode('utf-8', 'ignore')
            else:
                return None
        except httpx.TimeoutException:
            print(f"[Fetch] Timeout fetching {url} after {REQUEST_TIMEOUT}s")
        except httpx.TooManyRedirects:
             print(f"[Fetch] Too many redirects fetching {url}")
        except httpx.RequestError as e:
            if "Connection refused" not in str(e) and "Connect call failed" not in str(e):
                 print(f"[Fetch] Request error fetching {url}: {type(e).__name__}")
        except httpx.HTTPStatusError as e:
             if e.response.status_code != 404:
                  print(f"[Fetch] HTTP error fetching {url}: Status {e.response.status_code}")
        except Exception as e:
            print(f"[Fetch] Unknown error fetching {url}: {type(e).__name__} - {e}")
            traceback.print_exc()
        return None

def extract_links(html_content: str, base_url: str) -> Set[str]:
    """Extracts and resolves absolute links from HTML content."""
    links = set()
    if not html_content: return links
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href or href.startswith("#") or href.lower().startswith("mailto:") or href.lower().startswith("javascript:"):
                continue
            absolute_url = urljoin(base_url, href)
            parsed_url = urlparse(absolute_url)
            if parsed_url.scheme in ["http", "https"]:
                cleaned_url = parsed_url._replace(fragment="").geturl()
                links.add(cleaned_url)
    except Exception as e:
        print(f"[LinkExtract] Error parsing links from {base_url}: {type(e).__name__} - {e}")
    return links

async def should_queue_url(
    url: str,
    current_depth: int,
    max_depth: int,
    base_path: str,
    visited_urls: Set[str],
) -> Tuple[bool, str]:
    """Determines if a URL should be *queued for crawling*."""
    reason_prefix = f"(d={current_depth})"
    if url in visited_urls: return False, f"[{reason_prefix}] Visited"
    if current_depth > max_depth: return False, f"[{reason_prefix}] ❌ Exceeds max depth ({max_depth})"
    if not url.startswith(base_path): return False, f"[{reason_prefix}] ❌ Outside base path"
    if not url.startswith(("http://", "https://")): return False, f"[{reason_prefix}] ❌ Not HTTP/HTTPS"
    return True, ""

async def should_process_content(
    html_content: str,
    url: str,
    topic_embedding: Optional[List[float]],
) -> Tuple[bool, Optional[str]]:
    """Determines if the *content* should be processed based on semantic similarity."""
    if not CONTENT_PROCESSOR_AVAILABLE:
        print(f"[Content Filter] Skipping for {url} - Content processor functions not available.")
        cleaned_text_on_skip = clean_html_content(html_content) if html_content else None
        return True, cleaned_text_on_skip

    cleaned_text = clean_html_content(html_content)
    if not cleaned_text:
        return False, None

    semantic_passed = False
    content_sem_score = 0.0
    semantic_enabled = CONTENT_SEMANTIC_THRESHOLD <= 1.0

    if topic_embedding is not None and semantic_enabled:
        content_embedding = await get_embedding(cleaned_text)
        if content_embedding:
            content_sem_score = cosine_similarity(content_embedding, topic_embedding)
            semantic_passed = content_sem_score >= CONTENT_SEMANTIC_THRESHOLD
        else:
            print(f"[Content Filter] Warning: Failed to generate content embedding for {url}. Semantic check failed.")
            semantic_passed = False
    elif not semantic_enabled:
        semantic_passed = True
        content_sem_score = -1.0 # Indicate semantic disabled
    else: # No topic embedding provided, but semantic check might be enabled (e.g., threshold 0.3)
        # If no topic embedding, cannot perform semantic check, so effectively pass
        semantic_passed = True
        content_sem_score = -2.0 # Indicate no topic embedding


    should_process = semantic_passed
    sem_thresh_str = 'Disabled' if not semantic_enabled else f"{CONTENT_SEMANTIC_THRESHOLD:.2f}"
    # Adjust score string based on status
    if content_sem_score == -1.0:
        sem_score_str = "N/A (Sem Disabled)"
    elif content_sem_score == -2.0:
        sem_score_str = "N/A (No Topic Emb)"
    else:
        sem_score_str = f"{content_sem_score:.2f}"


    if should_process:
        log_msg = f"✅ [Content Passed] (Score: {sem_score_str}, Threshold: {sem_thresh_str}) -> {url}"
        print(log_msg)
        return True, cleaned_text
    else:
        log_msg = f"❌ [Content Rejected] (Score: {sem_score_str}, Threshold: {sem_thresh_str}) -> {url}"
        print(log_msg)
        return False, None # Return None for cleaned_text if rejected

def save_raw_html(html_content: str, filepath: str):
    """Saves HTML content to a file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8", errors="ignore") as f:
             f.write(html_content)
    except OSError as e:
        print(f"[SaveError] OS Error saving raw HTML to {filepath}: {e}")
    except Exception as e:
        print(f"[SaveError] Unexpected error saving {filepath}: {type(e).__name__} - {e}")
        traceback.print_exc()

# --- Main Crawling Logic ---

async def crawl_site(
    start_url: str,
    max_depth: int,
    topic_embedding: Optional[List[float]] = None
) -> Dict[str, Dict[str, Any]]:
    """Crawls a website, applying semantic filters concurrently."""
    if SAVE_RAW:
        Path(OUTPUT_RAW_DIR).mkdir(parents=True, exist_ok=True)

    try:
        parsed_start_url = urlparse(start_url)
        if not parsed_start_url.scheme or not parsed_start_url.netloc:
            raise ValueError("Invalid start URL provided.")
        p = Path(parsed_start_url.path)
        parent_path_obj = p.parent
        parent_path = parent_path_obj.as_posix()
        if not parent_path.endswith('/'): parent_path += '/'
        base_path = f"{parsed_start_url.scheme}://{parsed_start_url.netloc}{parent_path}"
    except Exception as e:
        print(f"[CrawlSetup] Error setting up crawl: {type(e).__name__}. Cannot start crawl.")
        traceback.print_exc()
        return {}

    print(f"[Crawl] Starting crawl from: {start_url}")
    print(f"[Crawl] Max Depth: {max_depth}")
    print(f"[Crawl] Restricting crawl to PARENT directory: {base_path}")
    semantic_status = f">= {CONTENT_SEMANTIC_THRESHOLD:.2f}" if CONTENT_SEMANTIC_THRESHOLD <= 1.0 and topic_embedding else "Disabled"
    print(f"[Crawl] Topic Embedding for Content Filter: {'Provided' if topic_embedding else 'Not Provided'} (Similarity {semantic_status})")
    print(f"[Crawl] Save Raw HTML (post-content filter): {SAVE_RAW}")
    print(f"[Crawl] Concurrent Request Limit: {CONCURRENT_REQUEST_LIMIT}")
    print(f"[Crawl] Request Timeout: {REQUEST_TIMEOUT}s")
    print("-" * 40)

    queue = asyncio.Queue()
    queue.put_nowait((start_url, 0))
    visited_urls = set() # Tracks URLs for which a fetch task was *created*
    pending_fetch_tasks: Dict[asyncio.Task, Tuple[str, int]] = {}
    scraped_data: Dict[str, Dict[str, Any]] = {} # Stores data for pages that PASS content filter
    fetch_semaphore = asyncio.Semaphore(CONCURRENT_REQUEST_LIMIT)
    content_filtered_urls = set() # Tracks URLs that FAILED content filter
    urls_failed_fetch = set() # Tracks URLs that failed during fetch (timeout, error)

    PROCESS_BATCH_SIZE = CONCURRENT_REQUEST_LIMIT * 2 # How many fetch results to process at once
    fetch_results_buffer: List[Tuple[str, int, Optional[str]]] = [] # Holds results before content check

    async with httpx.AsyncClient(follow_redirects=True, timeout=REQUEST_TIMEOUT, verify=True) as session:
        crawl_loop_count = 0
        max_loops = 100000 # Safety break

        while crawl_loop_count < max_loops:
            crawl_loop_count += 1
            processed_in_loop = 0

            # --- Queue Processing: Add new fetch tasks ---
            while not queue.empty() and len(pending_fetch_tasks) < CONCURRENT_REQUEST_LIMIT:
                url, depth = queue.get_nowait()

                should_queue_flag, _ = await should_queue_url(
                    url, depth, max_depth, base_path, visited_urls
                )

                if not should_queue_flag:
                    continue

                # Avoid queueing if already pending fetch
                if url in [ctx[0] for ctx in pending_fetch_tasks.values()]:
                    continue

                visited_urls.add(url) # Add to visited ONLY when task is created
                task = asyncio.create_task(fetch_url(session, url, fetch_semaphore))
                pending_fetch_tasks[task] = (url, depth)

            # --- Task Completion Processing ---
            if not pending_fetch_tasks and queue.empty() and not fetch_results_buffer:
                 print("[Crawl] Queue, pending tasks, and buffer are empty. Exiting crawl loop.")
                 break # Exit loop if nothing left to fetch or process

            completed_fetch_tasks = []
            if pending_fetch_tasks:
                try:
                    # Wait briefly for any task to complete
                    done, _ = await asyncio.wait(pending_fetch_tasks.keys(), return_when=asyncio.FIRST_COMPLETED, timeout=0.1)
                    completed_fetch_tasks.extend(done)
                except asyncio.TimeoutError:
                    pass # No tasks completed in this interval

            # --- Collect results from completed fetch tasks ---
            for task in completed_fetch_tasks:
                if task not in pending_fetch_tasks: continue # Should not happen, but safe check
                original_url, original_depth = pending_fetch_tasks.pop(task)

                try:
                    html_content = task.result()
                    if html_content is None:
                        # Mark as failed fetch if fetch_url returned None explicitly
                        urls_failed_fetch.add(original_url)
                    # Add result (or None) to buffer regardless of success/failure
                    fetch_results_buffer.append((original_url, original_depth, html_content))
                except asyncio.CancelledError:
                    print(f"[TaskResult] Fetch task for {original_url} was cancelled.")
                    urls_failed_fetch.add(original_url)
                    fetch_results_buffer.append((original_url, original_depth, None)) # Add None to buffer
                except Exception as e:
                    # Any other exception during task result retrieval
                    print(f"[TaskResult] Fetch task for {original_url} raised exception: {type(e).__name__}")
                    urls_failed_fetch.add(original_url)
                    fetch_results_buffer.append((original_url, original_depth, None)) # Add None to buffer

            # --- Process buffer in batches ---
            # Check if buffer is full OR if fetching is done and buffer still has items
            buffer_ready_to_process = len(fetch_results_buffer) >= PROCESS_BATCH_SIZE
            fetching_finished_and_buffer_has_items = (queue.empty() and not pending_fetch_tasks and fetch_results_buffer)

            if buffer_ready_to_process or fetching_finished_and_buffer_has_items:
                current_batch_size = len(fetch_results_buffer)
                print(f"[Crawl] Processing batch of {current_batch_size} fetched pages for content checks...")
                process_start_time = time.perf_counter()

                content_check_tasks = []
                task_context = [] # Store context associated with each task
                items_to_process = list(fetch_results_buffer) # Copy buffer for processing
                fetch_results_buffer.clear() # Clear the buffer

                for url, depth, html in items_to_process:
                    # Only create check task if HTML was successfully fetched
                    if html is not None:
                        check_task = asyncio.create_task(should_process_content(
                            html, url, topic_embedding
                        ))
                        content_check_tasks.append(check_task)
                        # Store context including HTML for later use
                        task_context.append({'url': url, 'depth': depth, 'html': html})
                    else:
                        # If HTML is None (fetch failed/cancelled), no need to check content
                        # We already added it to urls_failed_fetch implicitly or explicitly
                        pass # No task created for this item

                content_check_results = []
                if content_check_tasks:
                    # Gather results of content checks (will include exceptions if tasks failed)
                    content_check_results = await asyncio.gather(*content_check_tasks, return_exceptions=True)

                processed_in_loop += len(items_to_process) # Count items attempted in this batch

                new_links_to_queue = []
                # Match results back to context - IMPORTANT: results are in order of task creation
                for i, result in enumerate(content_check_results):
                    context = task_context[i] # Context corresponds to the i-th task created
                    url = context['url']
                    depth = context['depth']
                    html = context['html'] # HTML is needed for link extraction

                    if isinstance(result, Exception):
                        print(f"[Content Check Error] Task for {url} failed: {type(result).__name__} - {result}")
                        content_filtered_urls.add(url) # Treat content check failure as filtered out
                        continue

                    # Result is a tuple: (should_process: bool, cleaned_text: Optional[str])
                    should_process, _ = result # We only need the boolean here

                    if should_process:
                        # Passed filter: Add to final data, save raw (if enabled), extract links
                        scraped_data[url] = {"html": html, "depth": depth}
                        if SAVE_RAW:
                            filename = build_raw_filename(url)
                            filepath = os.path.join(OUTPUT_RAW_DIR, filename)
                            save_raw_html(html, filepath)
                        # Extract links only if depth allows further crawling
                        if depth < max_depth:
                            new_links = extract_links(html, url)
                            for link in new_links:
                                # Basic check: must be within the starting directory path
                                if link.startswith(base_path):
                                    new_links_to_queue.append((link, depth + 1))
                    else:
                        # Failed filter: Add to filtered list
                        content_filtered_urls.add(url)

                # --- Batch Queueing ---
                queued_urls_count = 0
                # Get current snapshot of queue/pending to avoid race conditions during check
                queued_urls_snapshot = {item[0] for item in list(queue._queue)}
                fetching_urls_snapshot = {ctx[0] for ctx in pending_fetch_tasks.values()}

                for link_url, link_depth in new_links_to_queue:
                    # Check against already visited, current queue snapshot, pending fetches, and filtered lists
                    if link_url not in visited_urls and \
                       link_url not in queued_urls_snapshot and \
                       link_url not in fetching_urls_snapshot and \
                       link_url not in content_filtered_urls and \
                       link_url not in urls_failed_fetch:
                        # Add to queue. visited_urls will be updated when the task is created later.
                        queue.put_nowait((link_url, link_depth))
                        queued_urls_count += 1
                        queued_urls_snapshot.add(link_url) # Add to snapshot to prevent duplicates *within this batch*


                process_duration = time.perf_counter() - process_start_time
                print(f"[Crawl] Batch processing finished in {process_duration:.2f}s. Queued {queued_urls_count} new links.")

            # Add a small sleep if no batch was processed but tasks/queue still exist
            # This prevents high CPU usage in edge cases where tasks complete slowly
            if not processed_in_loop and (pending_fetch_tasks or not queue.empty()):
                await asyncio.sleep(0.05)

        # --- End of Crawl Loop ---
        if crawl_loop_count >= max_loops:
             print(f"[Crawl] Warning: Crawl loop reached maximum iterations ({max_loops}). Terminating.")

    # --- Cleanup: Cancel any remaining tasks ---
    if pending_fetch_tasks:
         print(f"[Crawl] Cancelling {len(pending_fetch_tasks)} outstanding fetch tasks...")
         for task in list(pending_fetch_tasks.keys()):
             if not task.done(): task.cancel()
         # Wait for cancellations to complete
         await asyncio.gather(*pending_fetch_tasks.keys(), return_exceptions=True)

    # --- Final Summary ---
    print("-" * 40)
    print(f"[Crawl] Crawling finished.")
    total_visited_or_queued = len(visited_urls) # URLs for which a fetch task was created
    passed_filter_count = len(scraped_data)     # URLs that passed content filter

    # *** ADDED LOG LINE ***
    print(f"[Crawl Summary] Pages Scanned (tasks created): {total_visited_or_queued}. Pages Passed Filter: {passed_filter_count}.")
    # **********************

    print(f"[Crawl] URLs Failing Fetch: {len(urls_failed_fetch)}")
    # Corrected Filtered Count: Use the set we maintained
    print(f"[Crawl] URLs Filtered by Content Check: {len(content_filtered_urls)}")
    print(f"[Crawl] Final Pages Kept for Processing: {passed_filter_count}")
    if SAVE_RAW: print(f"[Crawl] Raw HTML saved to: {OUTPUT_RAW_DIR}")
    print("-" * 40)
    return scraped_data

# --- Standalone Execution Logic ---
if __name__ == "__main__":
    print("--- Raw Scraper (Standalone Single Page Fetch Mode) ---")
    print(f"--- Using config: Timeout={REQUEST_TIMEOUT}s, Output Dir={OUTPUT_RAW_DIR} ---")
    target_url = ""
    while not target_url:
        try:
            url_input = input("Enter the exact URL of the single page to fetch: ").strip()
            if not url_input: print("URL cannot be empty."); continue
            parsed = urlparse(url_input)
            if parsed.scheme in ["http", "https"] and parsed.netloc: target_url = url_input
            else: print("Invalid URL. Please include http:// or https://.")
        except EOFError: print("\nOperation cancelled."); sys.exit(0)
        except KeyboardInterrupt: print("\nOperation cancelled."); sys.exit(0)

    async def fetch_and_save_single_page(url: str):
        Path(OUTPUT_RAW_DIR).mkdir(parents=True, exist_ok=True)
        print(f"[SingleFetch] Fetching {url}...")
        semaphore = asyncio.Semaphore(1)
        async with httpx.AsyncClient(follow_redirects=True, timeout=REQUEST_TIMEOUT) as session:
            html_content = await fetch_url(session, url, semaphore)
        if html_content is not None:
            filename = build_raw_filename(url)
            filepath = os.path.join(OUTPUT_RAW_DIR, filename)
            save_raw_html(html_content, filepath)
            print(f"[SingleFetch] Successfully fetched and saved {url}\n              Output: {filepath}")
        else:
            print(f"[SingleFetch] Failed to fetch or process {url}. Check URL and network.")

    start_time = time.perf_counter()
    try:
        if sys.platform == "win32":
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(fetch_and_save_single_page(target_url))
    except KeyboardInterrupt: print("\n[SingleFetch] Operation cancelled.")
    except Exception as e: print(f"\n[SingleFetch Error] An unexpected error occurred: {type(e).__name__} - {e}"); traceback.print_exc()
    finally: end_time = time.perf_counter(); print(f"[SingleFetch] Finished in {end_time - start_time:.2f} seconds."); print("-" * 60)