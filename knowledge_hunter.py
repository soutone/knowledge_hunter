import requests
import tiktoken
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import os
import hashlib
import json

load_dotenv(override=True)

# === 1. Setup OpenAI Client ===
def get_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            with open("my_secrets", "r", encoding="utf-8") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            print("ERROR: OpenAI API key not found.")
            exit(1)
    return api_key

openai_client = OpenAI(api_key=get_api_key())

# === 2. Constants ===
DEFAULT_MAX_DEPTH = 2
VALID_MD_EXTENSIONS = (".md", ".markdown", ".mdx")
TOKEN_CHUNK_SIZE = 10000
FILENAME_CACHE_PATH = "filename_cache.json"
OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
filename_cache = json.load(open(FILENAME_CACHE_PATH)) if os.path.exists(FILENAME_CACHE_PATH) else {}

# === 3. Tokenizer ===
def get_tokenizer_for_model(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

# === 4. Scrapers ===
def scrape_github_markdown(url, visited=None, depth=0, max_depth=DEFAULT_MAX_DEPTH):
    if visited is None:
        visited = set()
    if depth > max_depth:
        return ""

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return ""
        html = r.text
    except Exception as e:
        print(f"[GitHub] Error accessing {url}: {e}")
        return ""

    visited.add(url)
    print(f"[GitHub] Crawling ({depth}): {url}")
    soup = BeautifulSoup(html, "html.parser")

    if "/blob/" in url and url.endswith(VALID_MD_EXTENSIONS):
        article = soup.find("article", {"class": "markdown-body"})
        if article:
            return article.get_text(separator="\n")

    contents = []
    for link in soup.find_all("a", href=True):
        abs_url = urljoin("https://github.com", link["href"])
        if abs_url not in visited:
            if "/blob/" in abs_url and abs_url.endswith(VALID_MD_EXTENSIONS):
                contents.append(scrape_github_markdown(abs_url, visited, depth + 1, max_depth))
            elif "/tree/" in abs_url:
                contents.append(scrape_github_markdown(abs_url, visited, depth + 1, max_depth))

    return "\n".join(contents)

def scrape_website(url, visited=None, depth=0, max_depth=DEFAULT_MAX_DEPTH):
    if visited is None:
        visited = set()
    if depth > max_depth:
        return ""

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return ""
        html = r.text
    except Exception as e:
        print(f"[Site] Error accessing {url}: {e}")
        return ""

    visited.add(url)
    print(f"[Site] Crawling ({depth}): {url}")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()

    text_content = [soup.get_text(separator="\n")]
    domain = urlparse(url).netloc
    for link in soup.find_all("a", href=True):
        abs_url = urljoin(url, link["href"])
        if abs_url not in visited and urlparse(abs_url).netloc == domain:
            text_content.append(scrape_website(abs_url, visited, depth + 1, max_depth))

    return "\n".join(text_content)

# === 5. Chunking & Summarization ===
def chunk_text_by_tokens(text: str, tokenizer, max_tokens=TOKEN_CHUNK_SIZE):
    token_ids = tokenizer.encode(text)
    return [tokenizer.decode(token_ids[i:i + max_tokens]) for i in range(0, len(token_ids), max_tokens)]

def summarize_text(text_chunk: str, model_name: str):
    system_prompt = (
        "You are summarizing documentation or markdown instructions to feed into ChatGPT. "
        "Keep function signatures, code snippets, and usage examples intact in triple backticks. "
        "Shorten extraneous text but don't remove essential detail or code."
    )

    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_chunk}
            ],
            temperature=0.0,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Summarize] Error summarizing chunk: {e}")
        return ""

# === 6. Filename Generation with Cache ===
def make_cache_key(url, content):
    return hashlib.sha256((url.lower().strip() + hashlib.sha256(content.encode("utf-8")).hexdigest()[:10]).encode("utf-8")).hexdigest()

def generate_descriptive_filename(content: str, url: str, model_name: str) -> str:
    key = make_cache_key(url, content)
    if key in filename_cache:
        return filename_cache[key]

    domain = urlparse(url).netloc.lower()
    prompt = f"""
    Based on the following content scraped from {url}, generate a short, descriptive filename (without extension) 
    that captures the main topic. Use 3-5 words with underscores instead of spaces. No file extension.

    Content sample:
    {content[:5000]}

    Output just the filename.
    """

    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You generate concise, descriptive filenames based on content."},
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

# === 7. Main ===
def main():
    model_name = "gpt-4o-2024-08-06"
    max_chunk_tokens = TOKEN_CHUNK_SIZE

    start_url = input("Enter the URL to scrape (default: https://github.com/openai/openai-python): ").strip() or "https://github.com/openai/openai-python"
    max_depth_input = input(f"Enter max crawl depth (default: {DEFAULT_MAX_DEPTH}): ").strip()
    max_depth = int(max_depth_input) if max_depth_input.isdigit() else DEFAULT_MAX_DEPTH

    print(f"[Main] Starting crawl from: {start_url} (depth={max_depth}) using model {model_name}")
    tokenizer = get_tokenizer_for_model(model_name)

    domain = urlparse(start_url).netloc.lower()
    all_text = scrape_github_markdown(start_url, max_depth=max_depth) if "github.com" in domain else scrape_website(start_url, max_depth=max_depth)

    if not all_text.strip():
        print("[Main] No content scraped. Exiting.")
        return

    print("[Main] Chunking text...")
    chunks = chunk_text_by_tokens(all_text, tokenizer, max_tokens=max_chunk_tokens)
    print(f"[Main] {len(all_text)} characters => {len(chunks)} chunk(s).")

    summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[Main] Summarizing chunk {i}/{len(chunks)}")
        summaries.append(summarize_text(chunk, model_name))

    final_summary = "\n\n".join(summaries)
    output_path = os.path.join(OUTPUT_FOLDER, generate_descriptive_filename(final_summary, start_url, model_name) + ".txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_summary)

    print(f"[Main] Summary saved to {output_path}")

if __name__ == "__main__":
    main()
