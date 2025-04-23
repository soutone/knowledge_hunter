# knowledge_hunter.py

import asyncio
import os
import sys
import time
import re
import json # Added for quality assessment saving
from urllib.parse import urlparse, unquote
from pathlib import Path
from typing import Optional, Dict, Any, List
import httpx
from bs4 import BeautifulSoup
import traceback

# --- Import Configuration ---
import config

# --- Import necessary functions ---
from raw_scraper import crawl_site
from content_processor import (
    process_scraped_data,
    clean_html_content, # Needed for cleaning during inference # [cite: 171]
    get_embedding,
    get_llm_response,   # <<< Needed for LLM-based inference
    count_tokens,       # <<< Needed for truncating text for LLM
    get_tokenizer,      # <<< Needed for count_tokens
    async_openai_client as openai_client, # [cite: 172]
)
from compiler import compile_txt_files # [cite: 172]

# --- Constants from Config ---
OUTPUT_DIR = config.OUTPUT_DIR # [cite: 29]
OUTPUT_RAW_DIR = config.OUTPUT_RAW_DIR # [cite: 30]
MAX_DEPTH = config.MAX_DEPTH # [cite: 15]
# HIERARCHICAL CHANGE: Read chosen models potentially set during processing
# These might be overridden later by user choice in process_scraped_data
EXTRACTION_MODEL = config.EXTRACTION_MODEL # [cite: 18]
CONSOLIDATION_MODEL = config.CONSOLIDATION_MODEL # [cite: 19]
RATING_MODEL = config.RATING_MODEL # Used for topic inference LLM call & final rating # [cite: 20, 172]
CONTENT_SEMANTIC_THRESHOLD = config.CONTENT_SEMANTIC_THRESHOLD # [cite: 25]
CHUNK_SIZE = config.CHUNK_SIZE_TOKENS # [cite: 22]
SKIP_COMPILATION = config.SKIP_COMPILATION # [cite: 17]
SAVE_RAW = config.SAVE_RAW # [cite: 16]

# IMPROVEMENT: Define Quality Assessment Output Directory
OUTPUT_QUALITY_DIR = os.path.join(config.BASE_DIR, 'output_quality') # [cite: 29]

# --- Helper Functions ---

def sanitize_for_filename(text: str) -> str:
    """Sanitizes a string for use in a filename component."""
    if not text: return "untitled"
    text = re.sub(r"^[a-zA-Z]+://", "", text) # [cite: 173]
    text = re.sub(r'[\\/*?:"<>|]+', "_", text) # [cite: 173]
    text = re.sub(r"[\s_]+", "_", text) # [cite: 173]
    max_base_len = 100
    return text.strip('_')[:max_base_len].lower()

def generate_output_filename(topic: str, domain: str, extension: str = ".txt") -> str:
    """Generates a descriptive filename for output files."""
    sanitized_topic = sanitize_for_filename(topic)
    sanitized_domain = sanitize_for_filename(domain.replace(':', '_')) # [cite: 174]
    max_len = 80
    filename_base = f"{sanitized_topic}_{sanitized_domain}"[:max_len] # [cite: 174]
    filename_base = re.sub(r"_+", "_", filename_base).strip('_') # [cite: 174]
    if not filename_base:
        filename_base = f"output_{int(time.time())}" # [cite: 174]
    return f"{filename_base}{extension}"

# --- Fetch single URL helper (for topic inference) ---
async def fetch_single_url_for_topic(url: str, timeout: float) -> Optional[str]:
    """Fetches HTML for a single URL, simplified for setup."""
    semaphore = asyncio.Semaphore(1) # Limit concurrency for setup fetch
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" # [cite: 174]
    }
    # Use a new client for this single request to avoid interactions with the main crawler's client state if any
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, verify=True) as session: # [cite: 175]
        async with semaphore:
            try:
                response = await session.get(url, headers=headers, timeout=timeout) # [cite: 175]
                response.raise_for_status() # Raise exception for 4xx/5xx status codes # [cite: 176]
                content_type = response.headers.get("content-type", "").lower() # [cite: 176]
                if "text/html" in content_type:
                    try:
                        return response.text # [cite: 176]
                    except UnicodeDecodeError: # [cite: 177]
                        print(f"[Setup Fetch Warning] Unicode decode error for {url}, trying utf-8 ignore.") # [cite: 177]
                        # Use response.content which holds the raw bytes
                        return response.content.decode('utf-8', 'ignore') # [cite: 177]
                else:
                    # print(f"[Setup Fetch Info] Non-HTML content type '{content_type}' for {url}")
                    return None # Return None for non-HTML content # [cite: 178]
            except httpx.TimeoutException:
                print(f"[Setup Fetch Error] Timeout fetching {url} after {timeout}s") # [cite: 178]
                return None
            except httpx.RequestError as e:
                # Catch broader request errors (DNS, connection refused, etc.)
                print(f"[Setup Fetch Error] Failed to fetch {url} for topic analysis: {type(e).__name__}") # [cite: 179]
                return None
            except httpx.HTTPStatusError as e: # [cite: 180]
                # Catch 4xx/5xx errors
                print(f"[Setup Fetch Error] HTTP error {e.response.status_code} fetching {url}") # [cite: 180]
                return None
            except Exception as e: # [cite: 180]
                # Catch any other unexpected errors during fetch
                print(f"[Setup Fetch Error] Unexpected error fetching {url}: {type(e).__name__}") # [cite: 181]
                traceback.print_exc() # Print traceback for unexpected errors # [cite: 181]
                return None

# --- Topic Inference Helper ---
async def infer_topic_from_url(start_url: str) -> Optional[str]: # [cite: 181]
    """
    Attempts to infer a topic focus using LLM analysis of content,
    falling back to H1/Title/Path, then URL structure.
    """ # [cite: 182]
    print(f"[Setup] Attempting to infer topic from: {start_url}...") # [cite: 183]
    inferred_topic: Optional[str] = None
    domain: Optional[str] = None
    parsed_url: Optional[Any] = None # Use Any to avoid potential urlparse typing issues across versions # [cite: 183]

    try:
        parsed_url = urlparse(start_url)
        domain = parsed_url.netloc
        if not domain:
             print("[Setup] Warning: Could not parse domain from start URL.") # [cite: 183]
             return None
    except ValueError:
         print("[Setup] Warning: Invalid start URL format.") # [cite: 184]
         return None

    # --- Attempt 1: LLM Inference from Content ---
    html_content = await fetch_single_url_for_topic(start_url, config.REQUEST_TIMEOUT) # [cite: 184]
    if html_content and openai_client: # Check if client is available
        print("[Setup] Analyzing page content with LLM for topic inference...")
        cleaned_text = clean_html_content(html_content) # [cite: 184]
        if cleaned_text: # [cite: 185]
            # Truncate cleaned text to avoid excessive token usage for inference
            max_inference_tokens = 1500 # Adjust as needed
            tokenizer = get_tokenizer()
            token_count = count_tokens(cleaned_text, tokenizer) # [cite: 185]

            if token_count > max_inference_tokens:
                if tokenizer: # [cite: 185]
                    encoded = tokenizer.encode(cleaned_text, disallowed_special=()) # Allow all special tokens for count # [cite: 186]
                    truncated_text = tokenizer.decode(encoded[:max_inference_tokens]) # [cite: 186]
                else: # Fallback if tokenizer failed
                    truncated_text = cleaned_text[:max_inference_tokens * 4] # Rough estimate # [cite: 186]
                print(f"[Setup Debug] Truncated content for LLM inference ({max_inference_tokens} tokens).") # [cite: 187]
            else:
                truncated_text = cleaned_text

            system_message = "You are a helpful assistant specialized in analyzing web page content." # [cite: 187]
            prompt = f"""Analyze the beginning of the text content from the webpage: {start_url}

Content Snippet:
\"\"\"
{truncated_text}
\"\"\"

Based *only* on the provided snippet, what is the primary subject or topic of this documentation page?
Respond with ONLY a concise, descriptive topic title (ideally 3-7 words). Do not add explanations.
Example output: "Using Asyncio Streams" or "LangChain Agent Configuration"

Topic Title:""" # [cite: 187, 188, 189, 190]

            try:
                # Use the cheaper/faster RATING_MODEL for this setup task
                # *** TYPEERROR FIX: Use max_tokens_completion instead of max_tokens ***
                llm_topic = await get_llm_response(
                    prompt=prompt, # [cite: 190]
                    system_message=system_message,
                    model=config.RATING_MODEL, # Use RATING_MODEL from config # [cite: 191]
                    temperature=0.1,
                    max_tokens_completion=50 # Corrected parameter name # [cite: 191]
                )

                if llm_topic and isinstance(llm_topic, str) and 3 < len(llm_topic) < 100: # Basic validation # [cite: 192]
                    # Further clean potential LLM artifacts like quotes
                    llm_topic = llm_topic.strip().strip('"').strip("'").strip() # [cite: 192]
                    if llm_topic: # [cite: 192]
                        inferred_topic = llm_topic # [cite: 193]
                        print(f"[Setup] Inferred topic from LLM analysis: '{inferred_topic}'") # [cite: 193]
                # elif llm_topic: # Log if response was received but invalid
                #     print(f"[Setup Debug] LLM response for topic invalid: '{llm_topic[:100]}...'") # [cite: 193]


            except Exception as llm_err:
                 # Print the actual error type and message
                 print(f"[Setup] LLM topic inference failed: {type(llm_err).__name__}") # [cite: 194]
                 # Optionally print traceback for debugging # [cite: 195]
                 # traceback.print_exc()
                 # Proceed to fallback methods
        else:
            print("[Setup] Content cleaned to empty, skipping LLM inference.") # [cite: 195]
    elif html_content and not openai_client:
        print("[Setup] OpenAI client not available, skipping LLM inference.") # [cite: 195]
    # --- End Attempt 1 --- # [cite: 195]

    # --- Attempt 2: Fallback to H1 / Title / Path ---
    if not inferred_topic and html_content: # [cite: 196]
        print("[Setup] Falling back to H1/Title/Path inference...")
        title_text: Optional[str] = None
        h1_text: Optional[str] = None
        path_text: Optional[str] = None
        try:
            soup = BeautifulSoup(html_content, "html.parser") # [cite: 196]
            # Extract Title
            title_tag = soup.find('title') # [cite: 197]
            if title_tag and title_tag.string:
                raw_title = title_tag.string.strip() # [cite: 197]
                # Try to remove common site names or separators at the end
                cleaned_title = re.sub(r'\s*([|-]|at|by)\s*([\w\s.-]+)$', '', raw_title, flags=re.IGNORECASE).strip() # [cite: 197]
                # If cleaning didn't change much or made it empty, try splitting
                if not cleaned_title or len(cleaned_title) > len(raw_title) - 2: # [cite: 198]
                    cleaned_title = re.split(r'[|-]', raw_title)[0].strip() # [cite: 198]
                if len(cleaned_title) > 3: title_text = cleaned_title # [cite: 198]
            # Extract H1
            h1_tag = soup.find('h1') # [cite: 199]
            if h1_tag:
                # Get text content, joining strings if multiple elements inside H1
                raw_h1 = ' '.join(h1_tag.stripped_strings) # [cite: 199]
                if raw_h1 and len(raw_h1) > 3: h1_text = raw_h1 # [cite: 199]
        except Exception as parse_e:
            print(f"[Setup] Error parsing H1/Title: {parse_e}") # [cite: 200]

        # Extract Path (can run even if HTML parsing failed)
        if parsed_url and parsed_url.path and parsed_url.path != '/': # [cite: 200]
            try:
                # Decode URL-encoded characters in path, strip slashes, split # [cite: 200]
                path_parts = unquote(parsed_url.path).strip('/').split('/') # [cite: 201]
                # Filter out common/generic terms and very short parts
                generic_terms = {'docs', 'documentation', 'api', 'reference', 'guide', # [cite: 201]
                                 'how-to', 'concepts', 'tutorials', 'index', 'html', 'htm', ''} # [cite: 201]
                meaningful_parts = [part.replace('-', ' ').replace('_', ' ')
                                    for part in path_parts
                                    if part.lower() not in generic_terms and len(part) > 2] # [cite: 202]
                if meaningful_parts: # [cite: 203]
                     # Join remaining parts and title-case them
                     path_text = ' '.join(meaningful_parts).title() # [cite: 203]
            except Exception as path_e:
                 print(f"[Setup] Error parsing path: {path_e}") # [cite: 203]

        # Combine H1/Title/Path, prioritizing H1, then Title, then Path
        topic_parts = []
        added_texts_lower = set() # Keep track to avoid duplicates

        if h1_text:
            topic_parts.append(h1_text) # [cite: 204]
            added_texts_lower.add(h1_text.lower())
        if title_text and title_text.lower() not in added_texts_lower: # [cite: 205]
            topic_parts.append(title_text) # [cite: 205]
            added_texts_lower.add(title_text.lower())
        if path_text and path_text.lower() not in added_texts_lower: # [cite: 205]
            topic_parts.append(path_text)
            # No need to add path_text to added_texts_lower as it's the last one checked

        if topic_parts:
            # Join with ' - ' and remove potential duplicate words from joining # [cite: 205]
            combined_topic = ' - '.join(topic_parts) # [cite: 206]
            # A simple way to remove duplicates while preserving order-ish
            combined_topic = ' '.join(dict.fromkeys(combined_topic.split())) # [cite: 206]

            if len(combined_topic) > 5: # Ensure it's reasonably long
                 inferred_topic = combined_topic # [cite: 206]
                 print(f"[Setup] Inferred topic from H1/Title/Path: '{inferred_topic}'") # [cite: 207]
    # --- End Attempt 2 ---


    # --- Attempt 3: Fallback to URL Structure ---
    if not inferred_topic and domain: # [cite: 207]
        print("[Setup] Falling back to URL structure inference.")
        try:
            domain_parts = domain.split('.')
            # Common subdomains or TLDs to ignore if they are the *only* part left # [cite: 207]
            common_subdomains = {'www', 'docs', 'dev', 'ai', 'developer', 'support', 'help', 'app', 'cloud', 'blog', 'info', 'com', 'org', 'net', 'io'} # [cite: 208]
            # Try using the part before the TLD first (e.g., 'google' from 'google.com')
            if len(domain_parts) >= 2 and domain_parts[-2].lower() not in common_subdomains: # [cite: 208]
                inferred_topic = domain_parts[-2].replace('-', ' ').title() # [cite: 209]
            # If that didn't work, try the first part if it's not common
            elif len(domain_parts) > 1 and domain_parts[0].lower() not in common_subdomains: # [cite: 209]
                 inferred_topic = domain_parts[0].replace('-', ' ').title() # [cite: 209]
            # Fallback to the whole domain if parts were too generic
            elif domain: # [cite: 209]
                 inferred_topic = domain.split('.')[0].replace('-', ' ').title() # Just take first part # [cite: 210]

            if inferred_topic:
                print(f"[Setup] Inferred topic from URL structure: '{inferred_topic}'") # [cite: 210]
            else:
                print("[Setup] Could not infer topic from URL structure.") # [cite: 210]
        except Exception as e:
             print(f"[Setup] Error during URL structure inference: {e}") # [cite: 211]

    # --- End Attempt 3 ---

    if not inferred_topic:
         print("[Setup] Failed to infer topic through all methods.") # [cite: 211]

    return inferred_topic
# --- END Topic Inference Helper ---


# --- Main Orchestration ---
async def main(start_url: str, topic_focus: str):
    """Main async function to run the knowledge hunter pipeline."""
    start_time = time.perf_counter() # [cite: 211]
    print(f"\n[Main] Starting Knowledge Hunter at {time.strftime('%Y-%m-%d %H:%M:%S')}") # [cite: 212]
    print("-" * 40)

    # Ensure output directories exist
    try:
        if SAVE_RAW:
            Path(OUTPUT_RAW_DIR).mkdir(parents=True, exist_ok=True) # [cite: 212]
            print(f"[Main] Ensured raw output directory exists: {os.path.abspath(OUTPUT_RAW_DIR)}") # [cite: 212]
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True) # [cite: 212]
        print(f"[Main] Ensured processed output directory exists: {os.path.abspath(OUTPUT_DIR)}") # [cite: 212]
        # IMPROVEMENT: Ensure quality assessment directory exists
        Path(OUTPUT_QUALITY_DIR).mkdir(parents=True, exist_ok=True)
        print(f"[Main] Ensured quality assessment directory exists: {os.path.abspath(OUTPUT_QUALITY_DIR)}")
    except OSError as e: # [cite: 212]
        print(f"[Main] Error creating output directories: {e}. Exiting.") # [cite: 213]
        traceback.print_exc() # [cite: 214]
        return

    # Calculate Domain
    try:
        parsed_url = urlparse(start_url) # [cite: 214]
        domain = parsed_url.netloc
        if not domain:
             raise ValueError("Could not parse domain from start URL.") # [cite: 214]
    except ValueError as e:
         print(f"[Main] Error: Invalid start URL '{start_url}': {e}. Exiting.") # [cite: 214]
         return # [cite: 215]
    except Exception as e:
        print(f"[Main] Error parsing URL '{start_url}': {type(e).__name__} - {e}. Exiting.") # [cite: 215]
        traceback.print_exc() # [cite: 215]
        return

    print(f"[Main] Using Topic Focus: '{topic_focus}'")

    # Log Configuration
    print("[Main] Using Configuration from config.py:") # [cite: 215]
    print(f"  - Start URL: {start_url}")
    print(f"  - Topic Focus: '{topic_focus}'")
    print(f"  - Domain: {domain}")
    print("-" * 20) # [cite: 215]
    print("  Scraping Config:") # [cite: 216]
    print(f"  - Max Depth: {MAX_DEPTH}") # [cite: 216]
    print(f"  - Concurrent Requests: {config.CONCURRENT_REQUEST_LIMIT}") # [cite: 216]
    print(f"  - Request Timeout: {config.REQUEST_TIMEOUT}s") # [cite: 216]
    print("-" * 20)
    print("  Content Filtering Config:") # [cite: 216]
    sem_thresh_str = 'Disabled' if CONTENT_SEMANTIC_THRESHOLD > 1.0 else f"{CONTENT_SEMANTIC_THRESHOLD:.2f}" # [cite: 216]
    print(f"  - Content Semantic Threshold: {sem_thresh_str}") # [cite: 216]
    print(f"  - Embedding Model: {config.EMBEDDING_MODEL}") # [cite: 216]
    print("-" * 20)
    print("  Processing Config:") # [cite: 216]
    # Use the actual models configured, potentially overridden later # [cite: 216]
    print(f"  - Extraction Model: {EXTRACTION_MODEL}") # [cite: 217]
    print(f"  - Consolidation Model: {CONSOLIDATION_MODEL}") # [cite: 217]
    print(f"  - Rating Model: {RATING_MODEL}") # [cite: 217]
    print(f"  - Chunk Size (Tokens): {CHUNK_SIZE}") # [cite: 217]
    print(f"  - LLM Concurrency Limit: {config.LLM_CONCURRENCY_LIMIT}") # [cite: 217]
    print("-" * 20)
    print("  Output Config:") # [cite: 217]
    print(f"  - Output Directory (Processed): {os.path.abspath(OUTPUT_DIR)}") # [cite: 217]
    print(f"  - Save Raw HTML: {SAVE_RAW}") # [cite: 217]
    if SAVE_RAW:
        print(f"  - Output Directory (Raw): {os.path.abspath(OUTPUT_RAW_DIR)}") # [cite: 217]
    # IMPROVEMENT: Log quality assessment dir
    print(f"  - Output Directory (Quality Assess): {os.path.abspath(OUTPUT_QUALITY_DIR)}")
    print(f"  - Skip Compilation: {SKIP_COMPILATION}") # [cite: 218]
    compiled_output_file_path = config.DEFAULT_COMPILED_OUTPUT_FILE # Use the configured default # [cite: 218]
    if not SKIP_COMPILATION:
        print(f"  - Compiled Output File: {os.path.abspath(compiled_output_file_path)}") # [cite: 218]
    print("-" * 40)

    # Initial Setup: Embeddings
    topic_embedding: Optional[List[float]] = None
    scraped_data: Dict[str, Dict[str, Any]] = {}

    openai_client_available = bool(openai_client) # [cite: 218]
    if not openai_client_available:
         print("[Setup] Warning: OpenAI client not initialized. LLM features disabled.") # [cite: 218]
         print("[Setup] Content semantic filtering and processing will be skipped.") # [cite: 219]

    content_semantic_enabled = CONTENT_SEMANTIC_THRESHOLD <= 1.0 # [cite: 219]
    need_embedding = content_semantic_enabled and topic_focus and openai_client_available # [cite: 219]

    if need_embedding:
        try:
            print(f"[Setup] Generating embedding for topic '{topic_focus}' (for content filtering)...") # [cite: 219]
            topic_embedding = await get_embedding(topic_focus, model=config.EMBEDDING_MODEL) # [cite: 219]
            if topic_embedding:
                 print("[Setup] Topic embedding generated successfully.") # [cite: 219]
            else:
                print("[Setup] Warning: Failed to generate topic embedding. Semantic filtering disabled.") # [cite: 220]
                content_semantic_enabled = False
        except Exception as e:
            print(f"[Setup] Error generating topic embedding: {type(e).__name__}. Semantic filtering disabled.") # [cite: 220]
            traceback.print_exc() # [cite: 221]
            topic_embedding = None
            content_semantic_enabled = False
    elif content_semantic_enabled and not topic_focus:
         print("[Setup] Warning: Semantic filtering enabled but no topic focus provided. Disabling semantic filter.") # [cite: 221]
         content_semantic_enabled = False # [cite: 222]
    elif content_semantic_enabled and not openai_client_available:
         print("[Setup] Warning: Semantic filtering enabled but OpenAI client unavailable. Disabling semantic filter.") # [cite: 222]
         content_semantic_enabled = False
    elif not content_semantic_enabled:
        print(f"[Setup] Skipping embedding generation: Content semantic threshold ({CONTENT_SEMANTIC_THRESHOLD}) > 1.0.") # [cite: 222]


    # --- Crawling ---
    print("-" * 40)
    print(f"[Main] Starting crawl from: {start_url}...") # [cite: 222]
    try: # [cite: 223]
        crawl_start_time = time.perf_counter() # [cite: 223]
        scraped_data = await crawl_site(
            start_url=start_url,
            max_depth=MAX_DEPTH,
            topic_embedding=topic_embedding if content_semantic_enabled else None # [cite: 223]
            # Pass other necessary config if crawl_site needs them
        )
        crawl_duration = time.perf_counter() - crawl_start_time # [cite: 223]
        print(f"[Main] Crawling completed in {crawl_duration:.2f} seconds.") # [cite: 224]
    except KeyboardInterrupt:
        print("\n[Main] Crawl interrupted by user.") # [cite: 224]
        if not scraped_data:
            print("[Main] No data collected before interruption. Exiting.") # [cite: 224]
            return # [cite: 225]
        print("[Main] Proceeding to process partially collected data...") # [cite: 225]
    except Exception as e:
        print(f"[Main] Error during crawling: {type(e).__name__} - {e}") # [cite: 225]
        traceback.print_exc()
        if not scraped_data:
            print("[Main] No data collected due to crawling error. Exiting.") # [cite: 225]
            return
        print("[Main] Proceeding to process potentially partial data due to crawling error...") # [cite: 226]

    # --- Processing ---
    print("-" * 40)
    consolidated_text : Optional[str] = None
    quality_rating : Optional[Dict[str, Any]] = None

    if not scraped_data:
        print("[Main] No pages available for processing.") # [cite: 226]
    elif not openai_client_available:
         print("[Main] Skipping content processing: OpenAI client unavailable.") # [cite: 226]
    else:
        print(f"[Main] Starting content processing for {len(scraped_data)} pages...") # [cite: 226]
        try:
            process_start_time = time.perf_counter() # [cite: 227]
            # process_scraped_data now returns chosen models if user selected them
            consolidated_text, quality_rating = await process_scraped_data(
                scraped_data=scraped_data,
                topic=topic_focus # [cite: 227]
                # Pass chosen_extraction_model, chosen_consolidation_model if needed # [cite: 228]
            )
            process_duration = time.perf_counter() - process_start_time # [cite: 228]
            print(f"[Main] Content processing finished in {process_duration:.2f} seconds.") # [cite: 228]
        except Exception as e:
            print(f"[Main] Error during content processing: {type(e).__name__} - {e}") # [cite: 228]
            traceback.print_exc() # [cite: 229]

    # --- Saving Output ---
    print("-" * 40)
    # Determine which models were actually used (read from config again,
    # as process_scraped_data might modify selections internally but doesn't return them)
    # In a more complex setup, process_scraped_data might return used model names.
    final_extraction_model = config.EXTRACTION_MODEL # Assume default unless changed # [cite: 229]
    final_consolidation_model = config.CONSOLIDATION_MODEL # Assume default unless changed # [cite: 230]

    if consolidated_text:
        # Use the config value for the final output file
        # output_filepath = os.path.abspath(config.DEFAULT_COMPILED_OUTPUT_FILE) # Original approach targeted compiler
        # Generate a *processed* filename based on topic/domain for the individual file if needed,
        # but the primary output seems intended for DEFAULT_COMPILED_OUTPUT_FILE potentially via compiler. # [cite: 230]
        # Let's save the consolidated output directly for now. # [cite: 231]
        processed_output_filename = generate_output_filename(topic_focus, domain) # [cite: 231]
        processed_output_filepath = os.path.join(OUTPUT_DIR, processed_output_filename) # [cite: 231]

        print(f"[Main] Saving consolidated documentation to: {processed_output_filepath}")
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(processed_output_filepath), exist_ok=True) # [cite: 231]
            with open(processed_output_filepath, "w", encoding="utf-8") as f:
                # Write Header # [cite: 232]
                f.write(f"Documentation for: {topic_focus} ({domain})\n") # [cite: 232]
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n") # [cite: 232]
                f.write(f"Source URL: {start_url}\n") # [cite: 232]
                f.write(f"Max Depth: {MAX_DEPTH}\n") # [cite: 232]
                f.write(f"Extraction Model: {final_extraction_model}\n") # [cite: 233]
                f.write(f"Consolidation Model: {final_consolidation_model}\n") # [cite: 233]
                cont_sem_thresh_str = 'Disabled' if not content_semantic_enabled else f"{CONTENT_SEMANTIC_THRESHOLD:.2f}" # [cite: 233]
                f.write(f"Content Filter Settings: Semantic Similarity={cont_sem_thresh_str}\n") # [cite: 233]
                f.write("="*80 + "\n\n") # [cite: 234]
                # Write Main Content
                f.write(consolidated_text) # [cite: 234]

                # IMPROVEMENT: Removed quality rating append from main file
                # if quality_rating:
                #     f.write("\n\n" + "="*80 + "\n")
                #     f.write("## Quality Assessment (Generated by AI)\n")
                #     f.write("="*80 + "\n\n")
                #     rating_val = quality_rating.get('rating_score', 'N/A')
                #     f.write(f"- **Rating Score (1-10):** {rating_val if rating_val is not None else 'N/A'}\n") # [cite: 235]
                #     f.write(f"- **Justification:**\n{quality_rating.get('rating_justification', 'N/A')}\n") # [cite: 236]
                #     f.write(f"- **Suggestions:**\n{quality_rating.get('rating_suggestions', 'N/A')}\n") # [cite: 236]
            print(f"[Main] Successfully saved documentation.") # [cite: 236]

            # --- IMPROVEMENT: Save Quality Assessment Separately ---
            if quality_rating:
                quality_filename = generate_output_filename(topic_focus, domain, extension="_quality_assessment.json") # Save as JSON for easier parsing
                quality_filepath = os.path.join(OUTPUT_QUALITY_DIR, quality_filename)
                print(f"[Main] Saving quality assessment to: {quality_filepath}")
                try:
                    # Ensure directory exists (might be redundant, but safe)
                    os.makedirs(os.path.dirname(quality_filepath), exist_ok=True)
                    with open(quality_filepath, "w", encoding="utf-8") as qf:
                        # Convert None score to string 'N/A' for JSON compatibility if needed, or handle in reading code
                        # quality_rating['rating_score'] = quality_rating.get('rating_score') if quality_rating.get('rating_score') is not None else 'N/A'
                        json.dump(quality_rating, qf, indent=2, ensure_ascii=False)
                    print("[Main] Successfully saved quality assessment.")
                except OSError as qe:
                     print(f"[Main] Error saving quality assessment to {quality_filepath}: {qe}")
                     traceback.print_exc()
                except Exception as qe_unexpected:
                     print(f"[Main] Unexpected error saving quality assessment: {type(qe_unexpected).__name__} - {qe_unexpected}")
                     traceback.print_exc()

            # --- Compilation Step ---
            # Compile *after* saving the individual processed file # [cite: 236]
            if not SKIP_COMPILATION: # [cite: 237]
                print("\n[Main] Starting final compilation step...") # [cite: 237]
                try:
                    # Compile files from OUTPUT_DIR into DEFAULT_COMPILED_OUTPUT_FILE
                    compile_txt_files(input_dir=OUTPUT_DIR, output_file=config.DEFAULT_COMPILED_OUTPUT_FILE) # [cite: 237]
                except Exception as e: # [cite: 237]
                    print(f"[Main] Error during compilation: {type(e).__name__} - {e}") # [cite: 238]
                    traceback.print_exc() # [cite: 238]
            else:
                print("\n[Main] Skipping final compilation step as per config.") # [cite: 238]

        except OSError as e: # [cite: 239]
            print(f"[Main] Error saving documentation to {processed_output_filepath}: {e}") # [cite: 239]
            traceback.print_exc() # [cite: 239]
        except Exception as e:
            print(f"[Main] Unexpected error saving documentation: {type(e).__name__} - {e}") # [cite: 239]
            traceback.print_exc()
    else:
        print("[Main] No consolidated text was generated to save.") # [cite: 239]
        if not SKIP_COMPILATION: # [cite: 240]
            print("[Main] Skipping compilation as no new content was generated.") # [cite: 240]


    # --- Finish ---
    end_time = time.perf_counter() # [cite: 240]
    print("-" * 40)
    print(f"[Main] Knowledge Hunter finished in {end_time - start_time:.2f} seconds.") # [cite: 240]
    print("-" * 40)

# --- Entry Point ---
if __name__ == "__main__":
    input_start_url: Optional[str] = None
    inferred_topic: Optional[str] = None
    final_topic_focus: Optional[str] = None

    # Handle Windows asyncio policy if needed before any async calls # [cite: 240]
    if sys.platform == "win32" and isinstance(asyncio.get_event_loop_policy(), asyncio.DefaultEventLoopPolicy): # [cite: 241]
         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()) # [cite: 241]

    print("--- Knowledge Hunter Setup ---")
    # Prompt for Start URL
    while not input_start_url:
        try:
            url_input = input("Enter the Start URL to scrape: ").strip() # [cite: 241]
            if not url_input:
                 print("Start URL cannot be empty.") # [cite: 241]
                 continue # [cite: 242]
            # Basic validation: requires scheme and netloc
            parsed = urlparse(url_input) # [cite: 242]
            if url_input and parsed.scheme in ["http", "https"] and parsed.netloc: # [cite: 242]
                input_start_url = url_input
            else: # [cite: 243]
                print("Invalid URL format. Please include http:// or https:// and a valid domain.") # [cite: 243]
        except (EOFError, KeyboardInterrupt): # [cite: 244]
            print("\nOperation cancelled by user during setup.")
            sys.exit(0) # [cite: 244]

    # Attempt Topic Inference BEFORE Prompting
    if input_start_url:
        try:
             # Run the async inference function
             inferred_topic = asyncio.run(infer_topic_from_url(input_start_url)) # [cite: 244]
        except RuntimeError as e: # [cite: 245]
             if "cannot be called from a running event loop" in str(e): # [cite: 245]
                 print("[Setup Error] Cannot run topic inference from a running event loop. Try running the script directly.") # [cite: 245]
                 # Attempt to get existing loop if available (e.g., in Jupyter)
                 loop = asyncio.get_event_loop() # [cite: 245]
                 if loop.is_running(): # [cite: 246]
                     print("[Setup Info] Running inference in existing loop...") # [cite: 246]
                     # Schedule the task and wait for it synchronously (use with caution)
                     task = loop.create_task(infer_topic_from_url(input_start_url)) # [cite: 246]
                     # This part is tricky and might block if not handled carefully
                     # For simplicity here, we might just skip inference in this case
                     print("[Setup Warning] Skipping topic inference due to running event loop conflict.") # [cite: 247]
                     inferred_topic = None # [cite: 248]
                 else:
                      print(f"[Setup Error] Runtime error during topic inference: {e}") # [cite: 248]
                      traceback.print_exc() # [cite: 249]
             else:
                 print(f"[Setup Error] Runtime error during topic inference: {e}") # [cite: 249]
                 traceback.print_exc() # [cite: 249]
        except Exception as infer_e:
            print(f"[Setup Error] Error during topic inference: {type(infer_e).__name__}. Proceeding without inference.") # [cite: 249]
            traceback.print_exc() # [cite: 250]

    # Prompt for Topic Focus
    while final_topic_focus is None:
        try:
            if inferred_topic:
                prompt_message = f"Inferred topic: '{inferred_topic}'. Press Enter to accept, or enter a different topic: " # [cite: 250]
            else:
                prompt_message = "Enter topic focus (e.g., 'Asyncio usage', required): " # [cite: 251]

            topic_input = input(prompt_message).strip() # [cite: 251]

            if topic_input: # User entered a topic
                final_topic_focus = topic_input # [cite: 251]
                print(f"[Setup] Using user-provided topic: '{final_topic_focus}'") # [cite: 252]
            elif inferred_topic: # User pressed Enter, accepting inferred topic
                final_topic_focus = inferred_topic # [cite: 252]
                print(f"[Setup] Accepted inferred topic: '{final_topic_focus}'") # [cite: 252]
            else: # User pressed Enter but nothing was inferred
                 print("Topic focus cannot be empty if not inferred. Please provide a topic.") # [cite: 252]
        except (EOFError, KeyboardInterrupt): # [cite: 253]
            print("\nOperation cancelled by user during setup.")
            sys.exit(0) # [cite: 253]

    # Run Main Logic
    if input_start_url and final_topic_focus: # [cite: 253]
        try:
            # Ensure policy is set again just before main run if needed # [cite: 253]
            # if sys.platform == "win32" and isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy): # [cite: 254]
            #      asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            asyncio.run(main(start_url=input_start_url, topic_focus=final_topic_focus)) # [cite: 254]
        except KeyboardInterrupt:
            print("\n[Main] Operation cancelled by user.") # [cite: 254]
        except RuntimeError as e:
             if "cannot be called from a running event loop" in str(e): # [cite: 255]
                 print("[Main Error] Cannot run main async logic from a running event loop.") # [cite: 255]
             else:
                 print(f"\n[Main Error] An unexpected runtime error occurred: {e}") # [cite: 255]
             traceback.print_exc() # [cite: 255]
        except Exception as e: # [cite: 256]
            print(f"\n[Main Error] An unexpected error occurred: {type(e).__name__} - {e}") # [cite: 256]
            traceback.print_exc() # [cite: 256]
    else:
        print("[Setup] Error: Missing Start URL or Topic Focus. Cannot proceed.") # [cite: 256]
        sys.exit(1) # [cite: 257]