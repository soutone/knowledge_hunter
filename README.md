# knowledge_hunter

**knowledge_hunter** is a Python script that automatically scrapes and summarizes documentation. It supports:

1. **GitHub Repos**  
   - Focuses on Markdown files (`.md`, `.markdown`, `.mdx`) instead of scraping entire codebases.  
   - Recursively follows repo directories up to a maximum depth.

2. **Regular Websites**  
   - Uses plain `requests` for same-domain links up to a maximum depth.

3. **Token-Based Chunking**  
   - Uses [tiktoken](https://github.com/openai/tiktoken) to chunk scraped text by tokens, matching your chosen model’s tokenizer.  
   - Prevents overshooting the model’s context window.
## Setup

1. **Install requirements**  
   ```bash
   pip install -r requirements.txt
   ```
2. **Add your OpenAI key**  
   - Create a file named `my_secrets` and paste your API key (e.g. `sk-...`) on a single line.

## Usage

1. **Edit `main()`** in `knowledge_hunter.py`:  
   - Change `model_name = "your-model-of-choice"` if desired.  
   - Adjust `start_url` to the docs or GitHub repo you want to summarize.
2. **Run**  
   ```bash
   python knowledge_hunter.py
   ```
3. **Output**  
   - The summarized text is saved to a file named `summarized_<something>.txt`, based on the URL.  

## Notes

- If you encounter Cloudflare blocks or login requirements, you may get empty results.  
- The token-based approach can also run into per-minute token limits. If so, **lower** `TOKEN_CHUNK_SIZE` in the script.  
- For **GitHub** repos, filenames are generated from `user_repo`. For other sites, it uses the domain name.  

Enjoy auto-summarizing your docs!