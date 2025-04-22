# -*- coding: utf-8 -*-
"""
Knowledge Hunter: Automated Documentation Extractor

Scrapes websites for technical documentation, filters relevant content using
keywords and semantic similarity, chunks text, extracts specific technical details
(functions, syntax) using an LLM, and compiles the results.
"""
import argparse # Keep argparse for potential future use or as reference
import asyncio
import hashlib
import json
import os
import re
import time
from collections import Counter
from urllib.parse import urljoin, urlparse, urlunparse
from pathlib import Path # Used for path manipulation

import httpx
import numpy as np
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

# --- Configuration ---
load_dotenv(override=True)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Please set it in your .env file or environment."
    )

openai_sync = OpenAI(api_key=OPENAI_KEY)
openai_async = AsyncOpenAI(api_key=OPENAI_KEY)

# --- Constants ---
DEFAULT_MAX_DEPTH = 2
FILENAME_CACHE_PATH = "filename_cache.json"
EMBEDDING_CACHE_PATH = "embedding_cache.json"
OUTPUT_FOLDER = "output"
DEFAULT_MODEL = "gpt-4o"
MODEL_CONTEXT_LIMITS = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384, # Often deprecated
    "gpt-4": 8192, # Often deprecated
    "gpt-4-turbo": 128000, # Context window size for gpt-4-turbo-preview etc.
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
}
AVAILABLE_MODELS = list(MODEL_CONTEXT_LIMITS.keys())

# --- Concurrency & Rate Limiting ---
MAX_HTTP_CONCURRENCY = 5
MAX_EMB_CONCURRENCY = 20
MAX_SUMMARY_CONCURRENCY = 5
http_sem = asyncio.Semaphore(MAX_HTTP_CONCURRENCY)
emb_sem = asyncio.Semaphore(MAX_EMB_CONCURRENCY)
summary_sem = asyncio.Semaphore(MAX_SUMMARY_CONCURRENCY)

# --- Similarity Threshold ---
SEMANTIC_SIMILARITY_THRESHOLD = 0.35

# --- Caching ---
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_json_cache(path):
    """Loads a JSON cache file if it exists."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[Cache] Error loading {path}: {e}. Starting with empty cache.")
            return {}
    return {}

def save_json_cache(path, cache_data):
    """Saves data to a JSON cache file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
    except IOError as e:
        print(f"[Cache] Error saving {path}: {e}")

filename_cache = load_json_cache(FILENAME_CACHE_PATH)
_emb_cache = load_json_cache(EMBEDDING_CACHE_PATH)

# --- Tokenization ---
def get_tokenizer_for_model(model_name=DEFAULT_MODEL):
    """Gets the appropriate tokenizer for a given model."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        print(f"[Tokenizer] Warning: No specific tokenizer found for {model_name}. Using cl100k_base.")
        return tiktoken.get_encoding("cl100k_base")

# --- Embeddings & Similarity ---
async def embed_text(text):
    """Generates an embedding for the given text, using a cache."""
    normalized_text = ' '.join(text.lower().strip().split())
    key = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

    if key in _emb_cache:
        return _emb_cache[key]

    async with emb_sem:
        try:
            resp = await openai_async.embeddings.create(
                model="text-embedding-3-small",
                input=normalized_text
            )
            vec = resp.data[0].embedding
            _emb_cache[key] = vec
            save_json_cache(EMBEDDING_CACHE_PATH, _emb_cache)
            return vec
        except Exception as e:
            print(f"[Embed] Error embedding text (first 50 chars): '{normalized_text[:50]}...': {e}")
            return None

def cosine(a, b):
    """Calculates the cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    try:
        a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / ((norm_a * norm_b) + 1e-9))
    except Exception as e:
        print(f"[Cosine] Error calculating similarity: {e}")
        return 0.0

# --- Keyword Extraction & Scoring ---
def extract_keywords(text, top_k=20):
    """Extracts the most common words (potential keywords) from text."""
    try:
        words = re.findall(r"\b[a-zA-Z0-9'-]{4,}\b", text.lower())
        stopwords = {"the", "and", "is", "in", "it", "to", "of", "a", "for", "with", "as", "on", "that", "this"}
        filtered_words = [word for word in words if word not in stopwords]
        return [w for w, _ in Counter(filtered_words).most_common(top_k)]
    except Exception as e:
        print(f"[Keywords] Error extracting keywords: {e}")
        return []

def score_link_quick(haystack, keywords):
    """Scores a link based on keyword presence."""
    haystack_lower = haystack.lower()
    score = sum(1 + haystack_lower.count(kw) // 2 for kw in keywords if kw in haystack_lower)
    return score

# --- Path Restriction Helper ---
def get_allowed_base_path(url_str):
    """
    Determines the allowed base path for crawling (up to one level above the start URL's directory).
    e.g., https://example.com/docs/product/category/page.html -> https://example.com/docs/product/
    e.g., https://example.com/docs/product/ -> https://example.com/docs/
    e.g., https://example.com/docs/ -> https://example.com/
    """
    parsed = urlparse(url_str)
    # Use pathlib for robust path manipulation
    p = Path(parsed.path)

    # If the path ends with a filename (has a suffix), get the parent dir first
    # Otherwise, treat the current path as the directory
    current_dir = p.parent if p.suffix else p

    # Get the parent of the current directory (one level up)
    parent_dir = current_dir.parent

    # Ensure the path ends with a slash
    allowed_path = str(parent_dir).replace('\\', '/') # Ensure forward slashes
    if not allowed_path.endswith('/'):
        allowed_path += '/'
    # Handle the root case separately
    if allowed_path == "//":
         allowed_path = "/"

    # Reconstruct the base URL
    base_url_parts = (parsed.scheme, parsed.netloc, allowed_path, '', '', '')
    return urlunparse(base_url_parts)

# --- Web Scraping ---
async def async_scrape_website(start_url, max_depth, topic_vec, seed_keywords):
    """Asynchronously scrapes a website starting from start_url."""
    visited = set()
    content_map = {}
    parsed_start_url = urlparse(start_url)
    domain = parsed_start_url.netloc
    # --- Calculate allowed base path ---
    allowed_base_path = get_allowed_base_path(start_url)
    print(f"[Async] Restricting crawl to base path: {allowed_base_path}")
    # -----------------------------------
    all_keywords = list(dict.fromkeys(seed_keywords))
    print(f"[Async] Initial keywords: {all_keywords}")

    async with httpx.AsyncClient(follow_redirects=True, http2=False, timeout=20.0) as client:

        async def crawl(url, depth):
            """Recursive crawl function."""
            nonlocal all_keywords

            if depth > max_depth or url in visited:
                return
            visited.add(url)

            async with http_sem:
                try:
                    print(f"[Async] Fetching ({depth}): {url}")
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                    }
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    html = resp.text
                except httpx.RequestError as e:
                    print(f"[Async] Network error fetching {url}: {e}")
                    return
                except httpx.HTTPStatusError as e:
                    print(f"[Async] HTTP error fetching {url}: {e.response.status_code} - {e.request.url}")
                    return
                except Exception as e:
                    print(f"[Async] Error fetching {url}: {e}")
                    return

            try:
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
                    tag.decompose()
                body_content = soup.body
                content = body_content.get_text("\n", strip=True) if body_content else ""

                if content:
                    content_map[url] = re.sub(r'\n{3,}', '\n\n', content).strip()
                    if depth == 0:
                         extracted_kws = extract_keywords(content)
                         all_keywords = list(dict.fromkeys(all_keywords + extracted_kws))
                         print(f"[Async] Updated keywords from first page: {all_keywords}")

            except Exception as e:
                print(f"[Async] Error parsing {url}: {e}")
                content_map[url] = ""


            tasks = []
            processed_links = set()
            scope = soup.body or soup
            if scope:
                for a in scope.find_all("a", href=True):
                    try:
                        href = a['href']
                        if not href or href.startswith(('#', 'javascript:', 'mailto:')):
                            continue

                        abs_url = urljoin(url, href)
                        parsed_abs_url = urlparse(abs_url)

                        # --- ADDED PATH CHECK ---
                        if (parsed_abs_url.netloc != domain or
                            abs_url in visited or
                            abs_url in processed_links or
                            parsed_abs_url.scheme not in ('http', 'https') or
                            not abs_url.startswith(allowed_base_path)): # Check if URL starts with the allowed base path
                            # Optional: Log why it was skipped
                            # if parsed_abs_url.netloc == domain and not abs_url.startswith(allowed_base_path):
                            #     print(f"[Filter]  पठान Skipping (path): {abs_url}")
                            continue
                        # -----------------------

                        if any(abs_url.lower().endswith(ext) for ext in ['.pdf', '.zip', '.jpg', '.png', '.gif', '.css', '.js']):
                            continue

                        processed_links.add(abs_url)
                        anchor_text = a.get_text(" ", strip=True)[:150]
                        haystack = f"{abs_url} {anchor_text}"

                        quick_score = score_link_quick(haystack, all_keywords)
                        passed_quick = (quick_score >= 1)

                        next_depth = depth + 1
                        sim = 0.0
                        passed_semantic = False

                        if passed_quick and next_depth <= max_depth:
                            link_vec = await embed_text(haystack[:512])
                            if link_vec and topic_vec:
                                sim = cosine(topic_vec, link_vec)
                                passed_semantic = sim >= SEMANTIC_SIMILARITY_THRESHOLD

                        status_icon = "❓"
                        if passed_quick:
                            status_icon = "✅" if passed_semantic else "❌"
                        print(f"[Filter] {status_icon} (d={next_depth}) q={quick_score:02d} sim={sim:.2f} → {abs_url}")

                        if passed_semantic:
                            tasks.append(crawl(abs_url, next_depth))

                    except Exception as e:
                        print(f"[Async] Error processing link {a.get('href', 'N/A')} on page {url}: {e}")


            await asyncio.gather(*tasks)

        await crawl(start_url, depth=0)
        combined_content = "\n\n--- Page Separator ---\n\n".join(filter(None, content_map.values()))
        return combined_content

# --- Text Chunking ---
def get_dynamic_token_chunk_size(model_name=DEFAULT_MODEL, output_buffer_tokens=1500, safety_margin=500):
    """Calculates a dynamic chunk size based on model context limit."""
    context_limit = MODEL_CONTEXT_LIMITS.get(model_name, 128000)
    chunk_size = context_limit - output_buffer_tokens - safety_margin
    return max(500, min(chunk_size, 8000))

def chunk_text_by_tokens(text, tokenizer, max_tokens):
    """Chunks text into segments with a maximum token count."""
    if not text:
        return []
    tokens = tokenizer.encode(text, disallowed_special=())
    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
        start_idx = end_idx
    final_chunks = []
    for chunk in chunks:
        chunk_token_count = len(tokenizer.encode(chunk, disallowed_special=()))
        if chunk_token_count > max_tokens:
             sub_chunks = chunk.split('\n\n')
             current_sub_chunk = ""
             for sc in sub_chunks:
                 sc_tokens = len(tokenizer.encode(sc, disallowed_special=()))
                 current_sub_chunk_tokens = len(tokenizer.encode(current_sub_chunk, disallowed_special=()))
                 if current_sub_chunk and current_sub_chunk_tokens + sc_tokens > max_tokens:
                     final_chunks.append(current_sub_chunk)
                     current_sub_chunk = ""
                 if sc_tokens > max_tokens:
                     if current_sub_chunk:
                         final_chunks.append(current_sub_chunk)
                         current_sub_chunk = ""
                     sc_encoded = tokenizer.encode(sc, disallowed_special=())
                     for i in range(0, len(sc_encoded), max_tokens):
                         final_chunks.append(tokenizer.decode(sc_encoded[i:i+max_tokens]))
                 elif current_sub_chunk_tokens + sc_tokens <= max_tokens:
                     current_sub_chunk += ("\n\n" + sc) if current_sub_chunk else sc
                 else:
                     final_chunks.append(current_sub_chunk)
                     current_sub_chunk = sc
             if current_sub_chunk:
                 final_chunks.append(current_sub_chunk)
        elif chunk:
            final_chunks.append(chunk)
    return [c for c in final_chunks if c]


# --- Summarization / Extraction ---
async def summarize_chunk(chunk, model_name, retries=3, initial_delay=5):
    """Summarizes a single chunk of text using OpenAI, with retries."""
    sys_prompt = (
        "You are an expert technical documentation extractor. Your sole focus is "
        "to identify and extract specific technical details from the provided text. "
        "IGNORE general explanations, introductions, concepts, or narrative prose. "
        "EXTRACT ONLY the following:\n"
        "1.  **Functions/Methods/Commands:** List their exact names.\n"
        "2.  **Syntax/Signature:** Provide the full syntax, including parameters (with types if available), "
        "arguments, options, flags, and return types (if mentioned).\n"
        "3.  **Purpose:** Briefly state the purpose or description of each function/method/command.\n"
        "4.  **Code Examples:** Include any direct code examples demonstrating usage.\n\n"
        "PRESENT the output clearly. Use Markdown formatting. Use triple backticks (```) for all code blocks, "
        "syntax definitions, and examples. If no functions/methods/commands are found in the chunk, state 'No technical functions or syntax found in this chunk.'\n"
        "DO NOT summarize the text in a narrative way. Extract the requested details directly."
    )

    delay = initial_delay
    for attempt in range(retries):
        try:
            async with summary_sem:
                resp = await openai_async.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": chunk}
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                    n=1,
                    stop=None,
                )
            summary = resp.choices[0].message.content.strip()
            if summary and "No technical functions" not in summary:
                 return summary
            elif "No technical functions" in summary:
                 print("[Summarize] Chunk contained no technical details.")
                 return None
            else:
                 print(f"[Summarize] Warning: Received potentially empty summary for chunk starting with: {chunk[:50]}...")
                 return summary
        except Exception as e:
            print(f"[Summarize] Error on attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                print(f"[Retry] Waiting {delay}s before retrying summarization...")
                await asyncio.sleep(delay)
                delay *= 1.5
            else:
                print(f"[Summarize] Failed to summarize chunk after {retries} attempts: {chunk[:100]}...")
                return None

async def summarize_all_chunks(chunks, model_name):
    """Summarizes a list of text chunks concurrently."""
    tasks = [summarize_chunk(chunk, model_name) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    return [summary for summary in results if summary]

# --- Consolidation ---
def consolidate_summaries(summaries, model_name):
    """Consolidates multiple summaries into a single document."""
    if not summaries:
        return "No relevant technical information extracted."
    if len(summaries) == 1:
        return summaries[0]

    joined_summaries = "\n\n---\n\n".join(summaries)

    prompt = (
        "You are consolidating extracted technical documentation details (functions, syntax, examples) from multiple text chunks. "
        "Your task is to merge these details into a single, coherent document.\n"
        "1.  **Combine:** Group information related to the same function/method/command together.\n"
        "2.  **De-duplicate:** Remove redundant entries or examples for the same item.\n"
        "3.  **Structure:** Organize the information logically (e.g., alphabetically by function name, or grouped by module if possible).\n"
        "4.  **Format:** Maintain clear Markdown formatting, especially for code blocks (```).\n"
        "5.  **Preserve Detail:** Ensure ALL extracted technical specifications (names, syntax, parameters, purposes, examples) are preserved.\n"
        "DO NOT add introductory sentences, narrative explanations, or summaries of the overall topic. Focus ONLY on merging the provided technical details cleanly."
    )

    try:
        resp = openai_sync.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": joined_summaries}
            ],
            temperature=0.0,
            max_tokens=4000,
        )
        consolidated = resp.choices[0].message.content.strip()
        return consolidated if consolidated else joined_summaries
    except Exception as e:
        print(f"[Consolidate] Error consolidating summaries: {e}")
        print("[Consolidate] Returning un-consolidated summaries.")
        return joined_summaries

# --- Filename Generation ---
def make_cache_key(url):
    """Creates a cache key for a URL."""
    parsed = urlparse(url)
    normalized_url = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path}{parsed.query}{parsed.fragment}"
    return hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()

def generate_descriptive_filename(content, url, model_name):
    """Generates a descriptive filename based on content and URL."""
    key = make_cache_key(url)
    if key in filename_cache:
        return filename_cache[key]

    domain = urlparse(url).netloc.lower()
    safe_domain = re.sub(r'[^a-z0-9_.-]+', '', domain)

    try:
        first_lines = content.strip().split('\n', 5)[:5]
        potential_title = ""
        for line in first_lines:
            line = line.strip()
            if line and len(line) > 5 and len(line) < 100:
                 if not line.endswith(('.', '!', '?')):
                     potential_title = line
                     break
        if potential_title:
            sanitized_title = re.sub(r'\s+', '_', potential_title)
            sanitized_title = re.sub(r'[^a-zA-Z0-9_]+', '', sanitized_title).lower()
            filename_base = "_".join(sanitized_title.split('_')[:5])
            if filename_base:
                filename = f"{filename_base}_{safe_domain}"
                filename_cache[key] = filename
                save_json_cache(FILENAME_CACHE_PATH, filename_cache)
                print(f"[Filename] Generated filename (rule-based): {filename}")
                return filename
    except Exception as e:
        print(f"[Filename] Error in rule-based filename generation: {e}. Falling back to LLM.")

    prompt = f"""Generate a concise, descriptive filename (3-5 lowercase words, separated by underscores, no extension) based on the primary topic of the following technical documentation content scraped from {url}. Focus on the core subject (e.g., 'api_reference', 'function_syntax', 'database_connection').

Content sample:
{content[:2500]}"""

    try:
        print("[Filename] Using LLM to generate filename.")
        resp = openai_sync.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You generate concise, descriptive filenames (3-5 words, lowercase, underscores)."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=20,
        )
        filename_base = resp.choices[0].message.content.strip()
        filename_base = re.sub(r'\s+', '_', filename_base)
        filename_base = re.sub(r'[^a-z0-9_]+', '', filename_base).lower()
        filename_base = "_".join(filename_base.split('_')[:5])

        if not filename_base:
            raise ValueError("LLM returned empty filename base")
        filename = f"{filename_base}_{safe_domain}"
    except Exception as e:
        print(f"[Filename] LLM filename generation failed: {e}. Using generic filename.")
        timestamp = int(time.time())
        filename = f"doc_extract_{safe_domain}_{timestamp}"

    filename_cache[key] = filename
    save_json_cache(FILENAME_CACHE_PATH, filename_cache)
    print(f"[Filename] Generated filename: {filename}")
    return filename

# --- Main Execution ---
async def run_knowledge_hunter(start_url, max_depth, topic_text, model_name=DEFAULT_MODEL):
    """Main orchestration function."""
    if not start_url:
        print("Error: Start URL cannot be empty.")
        return

    print(f"[Main] Starting Knowledge Hunter for URL: {start_url}")
    print(f"[Main] Topic: '{topic_text}', Max Depth: {max_depth}, Model: {model_name}")

    print("[Main] Generating topic embedding...")
    topic_vec = await embed_text(topic_text)
    if topic_vec is None:
        print("[Main] Error: Failed to generate topic embedding. Cannot proceed.")
        return

    seed_keywords = [w.strip().lower() for w in topic_text.split() if len(w) > 3]
    print(f"[Main] Initial keywords from topic: {seed_keywords}")

    print(f"[Main] Starting crawl from: {start_url}...")
    start_time = time.time()
    all_text = await async_scrape_website(
        start_url,
        max_depth=max_depth,
        topic_vec=topic_vec,
        seed_keywords=seed_keywords
    )
    crawl_time = time.time() - start_time
    print(f"[Main] Crawling completed in {crawl_time:.2f} seconds.")

    if not all_text or not all_text.strip():
        print("[Main] No relevant content scraped after filtering. Exiting.")
        return

    print(f"[Main] Total scraped text length: {len(all_text)} characters.")

    print("[Main] Chunking scraped text...")
    tokenizer = get_tokenizer_for_model(model_name)
    chunk_size = get_dynamic_token_chunk_size(model_name)
    print(f"[Main] Using chunk size: {chunk_size} tokens")
    chunks = chunk_text_by_tokens(all_text, tokenizer, max_tokens=chunk_size)
    print(f"[Main] Created {len(chunks)} chunks.")

    if not chunks:
        print("[Main] No text chunks generated. Exiting.")
        return

    print(f"[Main] Extracting technical details from {len(chunks)} chunks using {model_name}...")
    start_time = time.time()
    summaries = await summarize_all_chunks(chunks, model_name)
    summary_time = time.time() - start_time
    print(f"[Main] Extraction completed in {summary_time:.2f} seconds.")
    print(f"[Main] Successfully extracted details from {len(summaries)} chunks.")

    if not summaries:
        print("[Main] No technical details could be extracted from the scraped content. Exiting.")
        return

    print("[Main] Consolidating extracted details...")
    start_time = time.time()
    final_summary = consolidate_summaries(summaries, model_name)
    consolidation_time = time.time() - start_time
    print(f"[Main] Consolidation completed in {consolidation_time:.2f} seconds.")

    print("[Main] Generating filename and saving output...")
    filename_content_sample = final_summary[:500]
    output_filename_base = generate_descriptive_filename(filename_content_sample, start_url, model_name)
    output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename_base}.txt")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Source URL: {start_url}\n")
            f.write(f"Topic Focus: {topic_text}\n")
            f.write(f"Max Depth: {max_depth}\n")
            f.write(f"Model Used: {model_name}\n")
            f.write(f"Similarity Threshold: {SEMANTIC_SIMILARITY_THRESHOLD}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            f.write(final_summary)
        print(f"[Main] Final documentation saved to: {output_path}")
    except IOError as e:
        print(f"[Main] Error saving final output to {output_path}: {e}")

    try:
        compiler_path = os.path.join(os.path.dirname(__file__), 'compiler.py')
        if os.path.exists(compiler_path):
            import compiler
            print("[Main] Compiling all .txt files in output folder...")
            compiler.compile_txt_files()
        else:
            print("[Main] compiler.py not found, skipping compilation.")
    except ImportError:
        print("[Main] Could not import compiler module. Ensure compiler.py is present.")
    except Exception as e:
        print(f"[Main] Failed to run compiler: {e}")

    print("[Main] Saving caches...")
    save_json_cache(FILENAME_CACHE_PATH, filename_cache)
    save_json_cache(EMBEDDING_CACHE_PATH, _emb_cache)
    print("[Main] Knowledge Hunter finished.")


if __name__ == "__main__":
    start_url_input = ""
    while not start_url_input:
        start_url_input = input("Enter the Start URL to scrape: ").strip()
        if not start_url_input:
            print("URL cannot be empty.")

    max_depth_input = input(f"Enter max crawl depth (default: {DEFAULT_MAX_DEPTH}): ").strip()
    try:
        max_depth_val = int(max_depth_input) if max_depth_input else DEFAULT_MAX_DEPTH
    except ValueError:
        print(f"Invalid depth entered. Using default: {DEFAULT_MAX_DEPTH}")
        max_depth_val = DEFAULT_MAX_DEPTH

    topic_input = input("Enter topic focus (press Enter to infer from domain): ").strip()
    if not topic_input:
        try:
            topic_input = urlparse(start_url_input).netloc
            print(f"Inferring topic from domain: {topic_input}")
        except Exception:
             print("Could not infer topic from URL. Please provide a topic next time.")
             topic_input = "documentation"

    print(f"\nAvailable models: {', '.join(AVAILABLE_MODELS)}")
    model_input = input(f"Enter model name (default: {DEFAULT_MODEL}): ").strip()
    if not model_input or model_input not in AVAILABLE_MODELS:
        print(f"Using default model: {DEFAULT_MODEL}")
        model_input = DEFAULT_MODEL

    asyncio.run(run_knowledge_hunter(
        start_url=start_url_input,
        max_depth=max_depth_val,
        topic_text=topic_input,
        model_name=model_input
    ))