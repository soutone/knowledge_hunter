# knowledge_hunter

**knowledge_hunter** scrapes and summarizes documentation, making it easy to bring ChatGPT sessions **up to speed** with the latest docs—even if they were released **after** its knowledge cutoff. By feeding ChatGPT these summaries, you can get **accurate, context-aware** code generation and analysis for bleeding-edge tools and libraries.

## How It Works

1. **Scrapes Documentation**  
   - Supports both **GitHub Repos** and **regular websites**.  
   - For GitHub, it extracts **only Markdown files** (`.md`, `.markdown`, `.mdx`) instead of scraping entire codebases.  
   - For websites, it follows **same-domain links** up to a set depth.  

2. **Token-Based Chunking**  
   - Uses [tiktoken](https://github.com/openai/tiktoken) to split scraped text **precisely** into token-friendly chunks.  
   - Prevents overshooting ChatGPT’s context window.  

3. **ChatGPT-Friendly Summaries**  
   - Preserves **function signatures, code snippets, and usage examples** for direct reference.  
   - Summaries are structured to be **easily digestible** by ChatGPT for better code understanding and generation.  

## Setup

1. **Install requirements**  
   ```bash
   pip install -r requirements.txt
   ```
2. **Add your OpenAI key**  
   - Create a file named `my_secrets` and paste your API key (e.g. `sk-...`) on a single line.

## Usage

1. **Edit `main()`** in `knowledge_hunter.py`:  
   - Choose your model by setting `model_name = "your-model-of-choice"`.  
   - Adjust `start_url` to point to the docs or GitHub repo you want to summarize.  
2. **Run the script**  
   ```bash
   python knowledge_hunter.py
   ```
3. **Use the output in ChatGPT**  
   - The script saves a summarized file as `summarized_<something>.txt`.  
   - Upload this file into a ChatGPT session to help it **read and generate code** with the latest documentation.

## Notes

- **Cloudflare/login issues**: Some sites may block requests; in that case, try **finding an official GitHub repo** instead.  
- **Rate Limits**: If you hit OpenAI’s TPM (tokens per minute) limit, lower `TOKEN_CHUNK_SIZE` in the script.  
- **Filename Structure**:  
  - For GitHub repos, the output is named after the **user_repo** (e.g., `summarized_openai_openai-cookbook.txt`).  
  - For websites, it uses the **domain name**.  

By using **knowledge_hunter**, you can **extend ChatGPT’s capabilities beyond its knowledge cutoff**, making it an even more powerful coding assistant. 🚀