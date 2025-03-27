import asyncio
import httpx
import tiktoken
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import os
import hashlib
import json
import time
import re
from collections import Counter

load_dotenv(override=True)

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DEFAULT_MAX_DEPTH = 2
FILENAME_CACHE_PATH = "filename_cache.json"
OUTPUT_FOLDER = "output"
MODEL_CONTEXT_LIMITS = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4o": 128000,
    "gpt-4o-2024-08-06": 128000,
}

MAX_CONCURRENCY = 2
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
filename_cache = json.load(open(FILENAME_CACHE_PATH)) if os.path.exists(FILENAME_CACHE_PATH) else {}

def get_tokenizer_for_model(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

async def fetch_html(client, url):
    async with semaphore:
        try:
            resp = await client.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.text
        except Exception as e:
            print(f"[Async] Error fetching {url}: {e}")
    return ""

def extract_keywords(text: str, top_k: int = 20) -> list[str]:
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    common = Counter(words).most_common(top_k)
    return [word for word, _ in common]

def score_link(url: str, keywords: list[str]) -> int:
    return sum(kw in url.lower() for kw in keywords)

async def async_scrape_website(start_url, max_depth=DEFAULT_MAX_DEPTH):
    visited = set()
    domain = urlparse(start_url).netloc
    topic_keywords = []

    async with httpx.AsyncClient(follow_redirects=True) as client:
        async def crawl(url, depth):
            nonlocal topic_keywords

            if depth > max_depth or url in visited:
                return ""

            visited.add(url)
            html = await fetch_html(client, url)
            if not html:
                return ""

            print(f"[Async] Crawling ({depth}): {url}")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "header", "footer", "nav"]):
                tag.decompose()
            content = soup.get_text(separator="\n")

            if depth == 0:
                topic_keywords = extract_keywords(content)
                print(f"[Async] Topical keywords: {topic_keywords}")

            tasks = []
            for link in soup.find_all("a", href=True):
                abs_url = urljoin(url, link["href"])
                if urlparse(abs_url).netloc == domain and abs_url not in visited:
                    if depth == 0 or score_link(abs_url, topic_keywords) >= 1:
                        tasks.append(crawl(abs_url, depth + 1))

            sub_contents = await asyncio.gather(*tasks)
            return content + "\n" + "\n".join(sub_contents)

        return await crawl(start_url, depth=0)

def get_dynamic_token_chunk_size(model_name: str, output_buffer_tokens: int = 1000, safety_margin: int = 3000) -> int:
    context_limit = MODEL_CONTEXT_LIMITS.get(model_name, 8192)
    return context_limit - output_buffer_tokens - safety_margin

def chunk_text_by_tokens(text: str, tokenizer, max_tokens: int):
    paragraphs = text.split("\n\n")
    chunks = []
    buffer = []
    buffer_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        tokens = tokenizer.encode(para, disallowed_special=())

        if len(tokens) > max_tokens:
            for i in range(0, len(tokens), max_tokens):
                split_chunk = tokenizer.decode(tokens[i:i + max_tokens])
                chunks.append(split_chunk.strip())
            buffer = []
            buffer_tokens = 0
            continue

        if buffer_tokens + len(tokens) > max_tokens:
            chunks.append("\n\n".join(buffer).strip())
            buffer = [para]
            buffer_tokens = len(tokens)
        else:
            buffer.append(para)
            buffer_tokens += len(tokens)

    if buffer:
        chunks.append("\n\n".join(buffer).strip())

    # Final safety filter: ensure each chunk is under max_tokens
    safe_chunks = []
    for i, chunk in enumerate(chunks):
        tokens = tokenizer.encode(chunk, disallowed_special=())
        for j in range(0, len(tokens), max_tokens):
            sub_chunk = tokenizer.decode(tokens[j:j + max_tokens])
            safe_chunks.append(sub_chunk)
            print(f"[Chunking] Final chunk {len(safe_chunks)} has {len(tokens[j:j + max_tokens])} tokens")

    return safe_chunks



def summarize_text(text_chunk: str, model_name: str):
    system_prompt = (
        "You are summarizing highly technical documentation for developers."
        " Preserve all code blocks, commands, CLI tools, functions, configuration details, and specific terminology."
        " Wrap any code or commands in triple backticks."
        " Do not omit usage examples, flags, or syntax."
        " Summarize only redundant prose or introductory text."
    )

    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_chunk}
            ],
            temperature=0.0,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Summarize] Error summarizing chunk: {e}")
        return ""

def summarize_text_with_retry(text_chunk: str, model_name: str, max_retries: int = 5):
    delay = 10
    for attempt in range(max_retries):
        result = summarize_text(text_chunk, model_name)
        if result:
            return result
        print(f"[Retry] Waiting {delay}s before retrying...")
        time.sleep(delay)
        delay *= 1.5
    return ""

def consolidate_summaries(summaries, model_name):
    joined = "\n\n".join(summaries)
    consolidation_prompt = (
        "Consolidate the following technical summaries into a single, clear structure."
        " Ensure *no* technical detail is lost. Preserve ALL code examples, CLI commands, flags, and configurations."
        " Use clear Markdown structure if possible."
    )

    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": consolidation_prompt},
                {"role": "user", "content": joined}
            ],
            temperature=0.0,
            max_tokens=3000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Consolidate] Error consolidating summaries: {e}")
        return joined

def make_cache_key(url, content=None):
    return hashlib.sha256(url.lower().strip().encode("utf-8")).hexdigest()

def generate_descriptive_filename(content: str, url: str, model_name: str) -> str:
    key = make_cache_key(url)
    if key in filename_cache:
        return filename_cache[key]

    domain = urlparse(url).netloc.lower()
    prompt = f"""
    Based on the following content scraped from {url}, generate a short, descriptive filename (no extension).
    Use 3-5 lowercase words, separated by underscores. Do NOT include a file extension.

    Content sample:
    {content[:3000]}
    """

    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You generate concise, descriptive filenames."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=30
        )
        suggested = response.choices[0].message.content.strip()
        filename = f"{suggested}_{domain}"
        filename_cache[key] = filename
        with open(FILENAME_CACHE_PATH, "w") as f:
            json.dump(filename_cache, f, indent=2)
        return filename
    except:
        fallback = f"summarized_{domain}"
        filename_cache[key] = fallback
        with open(FILENAME_CACHE_PATH, "w") as f:
            json.dump(filename_cache, f, indent=2)
        return fallback

async def summarize_chunk(index, chunk, model_name):
    print(f"[Async] Summarizing chunk {index}")
    return await asyncio.to_thread(summarize_text_with_retry, chunk, model_name)

async def summarize_all_chunks(chunks, model_name):
    tasks = [summarize_chunk(i, chunk, model_name) for i, chunk in enumerate(chunks, 1)]
    return await asyncio.gather(*tasks)

def main():
    model_name = "gpt-4o-2024-08-06"
    max_chunk_tokens = get_dynamic_token_chunk_size(model_name)

    start_url = input("Enter the URL to scrape (default: https://openai.com): ").strip() or "https://openai.com"
    max_depth_input = input(f"Enter max crawl depth (default: {DEFAULT_MAX_DEPTH}): ").strip()
    max_depth = int(max_depth_input) if max_depth_input.isdigit() else DEFAULT_MAX_DEPTH

    print(f"[Main] Starting async crawl from: {start_url} (depth={max_depth}) using model {model_name}")
    tokenizer = get_tokenizer_for_model(model_name)

    all_text = asyncio.run(async_scrape_website(start_url, max_depth=max_depth))

    if not all_text.strip():
        print("[Main] No content scraped. Exiting.")
        return

    print("[Main] Chunking text...")
    start = time.time()
    chunks = chunk_text_by_tokens(all_text, tokenizer, max_tokens=max_chunk_tokens)
    print(f"[Main] Chunking complete. Took {time.time() - start:.2f}s")
    print(f"[Main] {len(all_text)} characters => {len(chunks)} chunk(s).")

    summaries = asyncio.run(summarize_all_chunks(chunks, model_name))

    if len(summaries) > 1:
        print("[Main] Consolidating summaries...")
        final_summary = consolidate_summaries(summaries, model_name)
    else:
        final_summary = summaries[0]

    output_path = os.path.join(OUTPUT_FOLDER, generate_descriptive_filename(final_summary, start_url, model_name) + ".txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_summary)

    print(f"[Main] Summary saved to {output_path}")

if __name__ == "__main__":
    main()