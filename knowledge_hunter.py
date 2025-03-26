import os
import requests
import tiktoken  # pip install tiktoken
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from openai import OpenAI

#########################################
# 1) READ API KEY FROM "my_secrets" FILE
#########################################
def get_api_key(filename="my_secrets"):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().strip()

#########################################
# 2) SETUP OPENAI CLIENT
#########################################
my_api_key = get_api_key()
openai_client = OpenAI(api_key=my_api_key)

#########################################
# 3) CONFIG / CONSTANTS
#########################################
MAX_DEPTH = 2
VALID_MD_EXTENSIONS = (".md", ".markdown", ".mdx")
TOKEN_CHUNK_SIZE = 10000  # e.g. for a 128k context model, leave overhead

#########################################
# 4) DYNAMIC TOKENIZER FETCH
#########################################
def get_tokenizer_for_model(model_name: str):
    """
    Attempt to load the official tiktoken tokenizer for a given model.
    If that fails, fallback to 'cl100k_base'.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        print(f"[get_tokenizer_for_model] WARNING: No built-in tokenizer for '{model_name}'. "
              "Falling back to 'cl100k_base'.")
        return tiktoken.get_encoding("cl100k_base")

#########################################
# 5) GITHUB SCRAPING (Markdown Only)
#########################################
def scrape_github_markdown(url, visited=None, depth=0):
    if visited is None:
        visited = set()
    if depth > MAX_DEPTH:
        return ""

    print(f"[scrape_github_markdown] Depth {depth}, URL: {url}")
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print(f"[scrape_github_markdown] Non-200 status: {r.status_code} at {url}")
            return ""
        html = r.text
    except Exception as e:
        print(f"[scrape_github_markdown] Request failed for {url}: {e}")
        return ""

    visited.add(url)
    soup = BeautifulSoup(html, "html.parser")

    # If this is a direct .md file link
    if "/blob/" in url.lower() and url.lower().endswith(VALID_MD_EXTENSIONS):
        article = soup.find("article", {"class": "markdown-body"})
        if article:
            return article.get_text(separator="\n")

    text_contents = []
    links = soup.find_all("a", href=True)
    for link in links:
        href = link["href"]
        abs_url = urljoin("https://github.com", href)
        if abs_url in visited:
            continue

        lower_href = href.lower()
        if "/blob/" in lower_href and any(lower_href.endswith(ext) for ext in VALID_MD_EXTENSIONS):
            text_contents.append(scrape_github_markdown(abs_url, visited, depth + 1))
        elif "/tree/" in lower_href:
            text_contents.append(scrape_github_markdown(abs_url, visited, depth + 1))

    return "\n".join(text_contents)

#########################################
# 6) NON-GITHUB SCRAPING (Requests Only)
#########################################
def scrape_website(url, visited=None, depth=0):
    """Recursively scrapes a normal website (up to MAX_DEPTH) using only requests."""
    if visited is None:
        visited = set()
    if depth > MAX_DEPTH:
        return ""

    print(f"[scrape_website] Depth {depth}, URL: {url}")
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print(f"[scrape_website] Non-200 status: {r.status_code} at {url}")
            return ""
        html = r.text
    except Exception as e:
        print(f"[scrape_website] Request failed for {url}: {e}")
        return ""

    visited.add(url)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()

    page_text = soup.get_text(separator="\n")
    text_content = [page_text]

    domain = urlparse(url).netloc
    links = soup.find_all("a", href=True)
    for link in links:
        abs_url = urljoin(url, link["href"])
        if abs_url not in visited and urlparse(abs_url).netloc == domain:
            text_content.append(scrape_website(abs_url, visited=visited, depth=depth + 1))

    return "\n".join(text_content)

#########################################
# 7) TOKEN-BASED CHUNKING
#########################################
def chunk_text_by_tokens(text: str, tokenizer, max_tokens=TOKEN_CHUNK_SIZE) -> list[str]:
    """
    Uses the given tokenizer to split the text into chunks of up to max_tokens.
    """
    token_ids = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(token_ids), max_tokens):
        chunk_slice = token_ids[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_slice)
        chunks.append(chunk_text)
    return chunks

#########################################
# 8) SUMMARIZATION
#########################################
def summarize_text(text_chunk: str, model_name: str):
    """
    Summarize the given text chunk with the specified model.
    """
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
        print(f"[summarize_text] OpenAI API error: {e}")
        return ""

#########################################
# 9) MAIN
#########################################
def main():
    # CHOOSE YOUR MODEL HERE:
    # e.g. "gpt-4o-2024-08-06", "gpt-3.5-turbo", "gpt-4", etc.
    model_name = "gpt-4o-2024-08-06"

    # For large context models, we can do a higher chunk size. 100k is just an example.
    # If you use a smaller model, 100k will be too big - you might want something like 2000-4000 tokens.
    max_chunk_tokens = TOKEN_CHUNK_SIZE

    # Example usage:

    # start_url = "https://github.com/openai/openai-python"
    start_url = "https://developers.google.com/drive/api/guides/about-sdk"

    print(f"[main] Starting scrape from: {start_url} with model {model_name}")

    # CREATE TOKENIZER FOR THE SELECTED MODEL
    tokenizer = get_tokenizer_for_model(model_name)

    parsed = urlparse(start_url)
    domain = parsed.netloc.lower()
    path = parsed.path.strip("/")
    all_text = ""

    # Distinguish GitHub vs non-GitHub
    if "github.com" in domain:
        print("[main] Detected GitHub domain; scraping only Markdown files.")
        if "/" in path:
            segments = path.split("/")
            if len(segments) >= 2:
                user_part, repo_part = segments[0], segments[1]
                custom_name = f"{user_part}_{repo_part}"
            else:
                custom_name = path.replace("/", "_")
        else:
            custom_name = path if path else domain

        all_text = scrape_github_markdown(start_url)
        output_base_name = f"summarized_{custom_name}"
    else:
        print("[main] Using simple requests-based approach.")
        all_text = scrape_website(start_url)
        output_base_name = f"summarized_{domain.replace(':','_')}"

    if not all_text.strip():
        print("[main] WARNING: No text was scraped. Possibly blocked or empty.")
    else:
        # Chunk text with the chosen tokenizer
        chunks = chunk_text_by_tokens(all_text, tokenizer, max_tokens=max_chunk_tokens)
        print(f"[main] Scraped {len(all_text)} characters => chunked into {len(chunks)} piece(s).")

        all_summaries = []
        for i, chunk in enumerate(chunks, start=1):
            print(f"[main] Summarizing chunk {i}/{len(chunks)} (length ~{len(chunk)} chars).")
            summary = summarize_text(chunk, model_name=model_name)
            all_summaries.append(summary)

        final_summary = "\n\n".join(all_summaries)
        output_filename = f"{output_base_name}.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(final_summary)

        print(f"[main] Done! Summaries saved to {output_filename}.")

if __name__ == "__main__":
    main()
