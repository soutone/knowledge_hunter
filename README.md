# knowledge_hunter

**knowledge_hunter** scrapes and summarizes online documentation, making it easy to bring ChatGPT sessions **up to speed** with the latest technical information—even if it was released **after** the language model's knowledge cutoff. By feeding ChatGPT these summaries, you can get **accurate, context-aware** code generation and analysis for bleeding-edge tools and libraries.

## How It Works

1.  **Scrapes Documentation**
    * Supports both **GitHub Repositories** and **regular websites**.
    * For **GitHub Repos**, it uses the **GitHub API** to find and fetch the raw content of Markdown files (`.md`, `.markdown`, `.mdx`), ensuring focused extraction of documentation. Requires a GitHub Personal Access Token.
    * For **websites**, it performs a standard crawl, following **same-domain links** up to a configurable depth (`MAX_DEPTH` in `config.py`).
    * Uses asynchronous requests (`httpx`, `asyncio`) for efficient fetching.

2.  **Content Filtering**
    * Applies **semantic similarity filtering** using OpenAI embeddings (`text-embedding-3-small` by default) to ensure scraped content is relevant to the user-provided topic focus. The threshold (`CONTENT_SEMANTIC_THRESHOLD`) is configurable.

3.  **Token-Based Chunking**
    * Uses [tiktoken](https://github.com/openai/tiktoken) to split scraped text **precisely** into token-friendly chunks (`CHUNK_SIZE_TOKENS` in `config.py`).
    * Prevents overshooting LLM context window limits during processing.

4.  **LLM-Powered Summarization & Analysis**
    * Processes relevant text chunks using specified OpenAI models (e.g., `gpt-4o`, configurable via `EXTRACTION_MODEL` and `CONSOLIDATION_MODEL`).
    * Extracts and preserves **key technical details**, including function signatures, code snippets, configuration examples, and step-by-step instructions.
    * Consolidates extracted details into a coherent summary.
    * Optionally generates a quality assessment report for the final summary using `RATING_MODEL`.

5.  **Output Generation**
    * Saves the final consolidated summary to a `.txt` file in the `output/` directory (e.g., `output/guide_for_manus_ai_agent_github_com.txt`).
    * Optionally compiles all generated `.txt` files from the `output/` directory into a single `documentation.txt` file (controlled by `SKIP_COMPILATION` in `config.py`).
    * Optionally saves raw scraped Markdown/HTML to `output_raw/` (`SAVE_RAW` in `config.py`).
    * Saves quality assessment reports (if generated) to `output_quality/`.

## Setup

1.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure API Keys:**
    * Create a file named `.env` in the project's root directory.
    * Add your OpenAI API key:
        ```dotenv
        OPENAI_API_KEY="sk-..."
        ```
    * **Add your GitHub Personal Access Token (PAT):**
        * Generate a PAT from your GitHub Developer settings ([GitHub Docs](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)). Grant it the `repo` scope (or `public_repo` if only scraping public repositories).
        * Add the token to your `.env` file:
            ```dotenv
            GITHUB_API_TOKEN="ghp_..."
            ```
        * *Note: The GitHub token is crucial for reliable scraping of repositories, as it significantly increases API rate limits.*

3.  **(Optional) Adjust Configuration:**
    * Modify settings like models, chunk size, concurrency limits, etc., in `config.py` as needed.

## Usage

1.  **Run the script:**
    ```bash
    python knowledge_hunter.py
    ```
2.  **Follow Prompts:**
    * The script will ask you to enter the **Start URL** (either a website page or a GitHub repository URL).
    * It will attempt to **infer the topic** using an LLM.
    * It will then ask you to confirm the inferred topic or provide your own **Topic Focus**. This topic is used for semantic filtering.

3.  **Use the Output:**
    * Find the generated summary `.txt` file in the `output/` directory. The filename is based on the topic and domain (e.g., `guide_for_manus_ai_agent_github_com.txt`).
    * If compilation is enabled (`SKIP_COMPILATION=False` in `config.py`), a combined `documentation.txt` file will also be created in the root directory.
    * Upload the relevant summary file into a ChatGPT session or use it as context for other LLM interactions to leverage the up-to-date documentation.

## Notes

* **GitHub Rate Limits:** Scraping GitHub repositories relies on the GitHub API. Using a `GITHUB_API_TOKEN` (set in `.env`) is highly recommended to avoid strict rate limits imposed on unauthenticated requests. You can adjust the API concurrency via `GITHUB_CONCURRENT_REQUEST_LIMIT` in `config.py`.
* **Website Scraping Issues:** Some websites may block scraping attempts (e.g., via Cloudflare or login requirements). If you encounter issues with a website, checking if an official GitHub repository exists for the documentation might be a better alternative.
* **OpenAI Rate Limits:** If you hit OpenAI’s TPM (tokens per minute) or RPM (requests per minute) limits during processing, consider:
    * Lowering `LLM_CONCURRENCY_LIMIT` in `config.py`.
    * Using a model with higher rate limits if available via your OpenAI plan.
    * Potentially reducing `CHUNK_SIZE_TOKENS` (though this increases the number of API calls).
* **Output Filenames:** Processed summaries are saved in the `output/` directory. Filenames are generated based on the sanitized topic focus and the source domain (e.g., `topic_focus_domain_com.txt`). The final compiled output (if enabled) is `documentation.txt`.

By using **knowledge_hunter**, you can efficiently gather and synthesize the latest technical documentation, extending the capabilities of LLMs beyond their training data cutoff. 🚀