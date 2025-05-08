# content_processor.py

import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
import traceback

# Added imports for .env loading
from dotenv import load_dotenv

import tiktoken
from bs4 import BeautifulSoup, Comment, NavigableString
from openai import AsyncOpenAI, OpenAIError
import numpy as np # Keep numpy for cosine_similarity

# --- Import Configuration ---
import config # Import hardcoded config values

# --- Environment Variable Loading ---
load_dotenv()

# --- Configuration from config.py ---
API_KEY = os.getenv("OPENAI_API_KEY")
EXTRACTION_MODEL_DEFAULT = config.EXTRACTION_MODEL # Default, may be overridden
CONSOLIDATION_MODEL_DEFAULT = config.CONSOLIDATION_MODEL # Default, may be overridden
RATING_MODEL = config.RATING_MODEL # Stays as configured for now
EMBEDDING_MODEL = config.EMBEDDING_MODEL
CHUNK_SIZE_TOKENS = config.CHUNK_SIZE_TOKENS
TIKTOKEN_ENCODING = config.TIKTOKEN_ENCODING
LLM_CONCURRENCY_LIMIT = config.LLM_CONCURRENCY_LIMIT

# --- OpenAI Client Initialization ---
async_openai_client: Optional[AsyncOpenAI] = None
if API_KEY:
    try:
        async_openai_client = AsyncOpenAI(api_key=API_KEY)
        print("[OpenAI Client] AsyncOpenAI client initialized successfully.")
    except Exception as e:
        print(f"[OpenAI Client Error] Could not initialize AsyncOpenAI client: {e}")
        traceback.print_exc()
else:
    print("[OpenAI Client Warning] OPENAI_API_KEY not found. LLM features disabled.")

# --- Semaphore for LLM Calls ---
llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY_LIMIT)

# --- Model Pricing (Approximate - Update as needed) ---
MODEL_CONTEXT_LIMITS = {
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-3.5-turbo-0125": 16385,
    "gpt-3.5-turbo": 16385, # Check specific variant if needed
}
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
}

def get_model_pricing(model_name: str) -> Dict[str, float]:
    """Retrieves pricing for a model, returns zeros if not found."""
    if model_name not in MODEL_PRICING:
        print(f"[Pricing Warning] Pricing for model '{model_name}' not found. Using $0.00.")
    return MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})

def get_model_context_limit(model_name: str) -> int:
    """Retrieves context limit for a model, returns a default if not found."""
    limit = MODEL_CONTEXT_LIMITS.get(model_name)
    if limit is None:
        print(f"[Context Limit Warning] Context limit for model '{model_name}' not found. Using default 100,000.")
        # Check common prefixes if exact match fails
        if model_name.startswith("gpt-4o-mini"): limit = MODEL_CONTEXT_LIMITS.get("gpt-4o-mini", 100000)
        elif model_name.startswith("gpt-4o"): limit = MODEL_CONTEXT_LIMITS.get("gpt-4o", 100000)
        elif model_name.startswith("gpt-4-turbo"): limit = MODEL_CONTEXT_LIMITS.get("gpt-4-turbo", 100000)
        elif model_name.startswith("gpt-3.5-turbo"): limit = MODEL_CONTEXT_LIMITS.get("gpt-3.5-turbo", 16000)
        else: limit = 100000 # Default fallback
    return limit

# --- Helper Functions ---
_tokenizer_cache = {}
def get_tokenizer(encoding_name: str = TIKTOKEN_ENCODING):
    """Gets the tiktoken tokenizer, using a simple cache."""
    if encoding_name in _tokenizer_cache:
        return _tokenizer_cache[encoding_name]
    try:
        tokenizer = tiktoken.get_encoding(encoding_name)
        _tokenizer_cache[encoding_name] = tokenizer
        return tokenizer
    except Exception as e:
        print(f"Error getting tokenizer {encoding_name}: {e}")
        _tokenizer_cache[encoding_name] = None
        return None

def count_tokens(text: str, tokenizer = None) -> int:
    """Counts tokens in a string using the specified or default tokenizer."""
    if tokenizer is None: tokenizer = get_tokenizer()
    if not tokenizer or not text: return len(text) // 4 if text else 0 # Rough estimate
    try:
        # Handle potential errors if the tokenizer fails on specific text patterns
        return len(tokenizer.encode(text, disallowed_special=()))
    except Exception as e:
        print(f"Warning: Token counting error - {e}. Falling back to estimation.")
        # Fallback estimate on error - split by spaces/newlines as a rough proxy
        return len(re.findall(r'\S+|\n', text))

# --- HTML Cleaning ---
def clean_html_content(html_content: str) -> str:
    """ Cleans HTML content, focusing on main text and basic structure """
    if not html_content:
        return ""
    try:
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove common non-content tags more aggressively
        tags_to_remove = ['script', 'style', 'nav', 'footer', 'aside', 'form', 'header', 'button', 'input', 'select', 'textarea', 'figure', 'figcaption', 'link', 'meta', 'noscript']
        for tag_name in tags_to_remove:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Remove elements by role or aria attributes indicating non-content
        attributes_to_check = {'role': ['navigation', 'banner', 'contentinfo', 'search', 'complementary', 'form', 'menu', 'menubar', 'toolbar'], 'aria-hidden': 'true'}
        for attr, values in attributes_to_check.items():
            if isinstance(values, list):
                for value in values:
                    for tag in soup.find_all(attrs={attr: value}): tag.decompose()
            else:
                for tag in soup.find_all(attrs={attr: values}): tag.decompose()

        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Try to find the main content area
        main_content = (soup.find('main') or soup.find('article') or soup.find('div', role='main') or
                        soup.find('div', id='content') or soup.find('div', class_='content') or
                        soup.find('div', class_='main-content') or soup.find('body'))
        if not main_content: main_content = soup # Fallback to whole soup if no better container found

        text_parts = []
        # Extract text primarily from paragraphs, headings, list items, code blocks, and table data
        # Preserve line breaks within <pre> tags
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'pre', 'td', 'th'], recursive=True):
            if isinstance(element, NavigableString): continue

            # Skip elements if they are descendants of already removed tags (should be handled by decompose, but double-check)
            is_irrelevant_parent = False
            for parent in element.parents:
                if parent.name in tags_to_remove:
                    is_irrelevant_parent = True
                    break
            if is_irrelevant_parent: continue

            if element.name == 'pre':
                # Preserve code block formatting somewhat
                code_text = element.get_text(strip=False) # Keep internal whitespace
                # Basic formatting for text output
                text_parts.append(f"\n---\nCODE BLOCK:\n{code_text.strip()}\n---\n")
            elif element.name in ['td', 'th']:
                 # Simple table cell extraction, separated by tabs within a row
                 text_parts.append(element.get_text(separator='\t', strip=True))
                 # Add newline after table row (approximation, relies on finding TR parent)
                 if element.find_parent('tr') and element == element.find_parent('tr').find_all(['td','th'])[-1]:
                    text_parts.append('\n')
            elif element.name.startswith('h'):
                level = int(element.name[1])
                marker = "#" * level # Use markdown for headings in temp cleaning
                text_parts.append(f"\n{marker} {element.get_text(strip=True)}\n")
            elif element.name == 'li':
                # Simple list item marker
                 text_parts.append(f"* {element.get_text(strip=True)}")
            else: # Primarily paragraphs ('p')
                text = element.get_text(strip=True)
                if text:
                    text_parts.append(text)

        # Join parts, clean up whitespace and multiple newlines
        cleaned_text = "\n".join(text_parts) # Use single newline join initially
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text) # Collapse excess newlines to max 2
        cleaned_text = re.sub(r' +\n', '\n', cleaned_text) # Remove trailing spaces before newline
        cleaned_text = re.sub(r'[ \t]{2,}', ' ', cleaned_text) # Replace multiple spaces/tabs with single space
        cleaned_text = re.sub(r'\n +\*', '\n*', cleaned_text) # Clean up list markers

        return cleaned_text.strip()

    except Exception as e:
        print(f"[CleanHTML] Error cleaning HTML: {type(e).__name__} - {e}")
        traceback.print_exc()
        # --- Fallback Mechanism ---
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            for tag in soup(['script', 'style']): tag.decompose()
            return soup.get_text(separator='\n', strip=True)
        except Exception as fallback_e:
            print(f"[CleanHTML] Fallback text extraction failed: {fallback_e}")
            return ""

# --- Chunking ---
def chunk_text(text: str, max_tokens: int = CHUNK_SIZE_TOKENS) -> List[str]:
    """ Splits text into chunks based on paragraphs and token limits. """
    tokenizer = get_tokenizer()
    if not text or not tokenizer: return [text] if text else []

    chunks = []
    current_chunk_parts = []
    current_tokens = 0
    force_split_threshold = int(max_tokens * 1.2) # Threshold to force split very long paragraphs

    # Split by double newlines first, preserving them somewhat
    # Consider code blocks as single paragraphs initially
    paragraphs = re.split(r'(\n---\nCODE BLOCK:.*?\n---\n|\n\n+)', text, flags=re.DOTALL)

    # Re-join paragraphs with their trailing double newlines or code blocks
    processed_paragraphs = []
    i = 0
    while i < len(paragraphs):
        part = paragraphs[i]
        if not part: i += 1; continue
        # Check if the next part is a separator/code block
        if i + 1 < len(paragraphs) and re.match(r'(\n---\nCODE BLOCK:.*?\n---\n|\n\n+)', paragraphs[i+1], flags=re.DOTALL):
            # Join the paragraph with its separator/code block
            processed_paragraphs.append(part + paragraphs[i+1])
            i += 2
        else:
            # Just the paragraph part
            processed_paragraphs.append(part)
            i += 1

    for para in processed_paragraphs:
        para = para.strip() # Strip leading/trailing whitespace from the paragraph itself
        if not para: continue

        para_tokens = count_tokens(para, tokenizer)
        separator_tokens = count_tokens("\n\n", tokenizer) if current_chunk_parts else 0

        # --- Force Split Long Paragraphs ---
        if para_tokens > force_split_threshold:
            # If current chunk has content, finalize it before handling the huge paragraph
            if current_chunk_parts:
                chunks.append("\n\n".join(current_chunk_parts))
                current_chunk_parts = []
            current_tokens = 0

            print(f"[Chunking] Warning: Paragraph ({para_tokens} tk) > force split threshold ({force_split_threshold} tk). Force splitting.")
            # Simple split by sentences or lines as fallback - might not be ideal
            sub_chunks = []
            current_sub_chunk = ""
            current_sub_tokens = 0
            # Try splitting by sentence-ending punctuation first
            sentences = re.split(r'(?<=[.!?])\s+', para)
            if len(sentences) <= 1: # If no sentences, try splitting by newline
                sentences = para.splitlines(keepends=True)

            for sentence in sentences:
                sentence_tokens = count_tokens(sentence, tokenizer)
                if current_sub_tokens + sentence_tokens <= max_tokens:
                    current_sub_chunk += sentence
                    current_sub_tokens += sentence_tokens
                else:
                    # Add the current sub-chunk if it's not empty
                    if current_sub_chunk: sub_chunks.append(current_sub_chunk.strip())
                    # Start a new sub-chunk, handle sentence > max_tokens
                    if sentence_tokens > max_tokens:
                        print(f"[Chunking] Warning: Sentence/Line within large paragraph ({sentence_tokens} tk) > max chunk size ({max_tokens} tk). Truncating sub-split.")
                        # Basic truncation for extremely long lines/sentences
                        encoded = tokenizer.encode(sentence)
                        sub_chunks.append(tokenizer.decode(encoded[:max_tokens]).strip())
                        current_sub_chunk = ""
                        current_sub_tokens = 0
                    else:
                        current_sub_chunk = sentence
                        current_sub_tokens = sentence_tokens

            if current_sub_chunk: sub_chunks.append(current_sub_chunk.strip())

            chunks.extend(sub_chunks) # Add all sub-chunks
            continue # Move to the next paragraph

        # --- Regular Chunking ---
        # If adding the new paragraph fits
        if current_tokens + para_tokens + separator_tokens <= max_tokens:
            current_chunk_parts.append(para)
            current_tokens += para_tokens + separator_tokens
        # If the paragraph itself is larger than max_tokens (but not force_split threshold)
        elif para_tokens > max_tokens:
             # Finalize the current chunk if it exists
             if current_chunk_parts:
                 chunks.append("\n\n".join(current_chunk_parts))
             print(f"[Chunking] Warning: Paragraph ({para_tokens} tk) > max chunk size ({max_tokens} tk). Placing in own chunk.")
             chunks.append(para) # Add the large paragraph as its own chunk
             current_chunk_parts = [] # Reset for the next chunk
             current_tokens = 0
        # If the paragraph doesn't fit in the current chunk, but is not oversized itself
        else:
            # Finalize the current chunk
            if current_chunk_parts: chunks.append("\n\n".join(current_chunk_parts))
            # Start a new chunk with the current paragraph
            current_chunk_parts = [para]
            current_tokens = para_tokens

    # Add the last remaining chunk
    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))

    # print(f"[Chunking] Split text into {len(chunks)} chunks (max_tokens={max_tokens}).") # Reduce log noise
    return chunks

# --- LLM Interactions ---
async def get_llm_response(
    prompt: str, system_message: str = "You are a helpful assistant.",
    model: str = EXTRACTION_MODEL_DEFAULT, temperature: float = 0.1,
    max_tokens_completion: Optional[int] = None, # Renamed from max_tokens to avoid clash
    retry_attempts: int = 2, initial_delay: float = 1.0,
) -> Optional[str]:
    """ Generic LLM call function """
    if not async_openai_client: print("[LLM Error] OpenAI client not initialized."); return None
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    delay = initial_delay
    async with llm_semaphore:
        for attempt in range(retry_attempts + 1):
            try:
                # print(f"[LLM Call Debug] Model: {model}, Temp: {temperature}, MaxCompTokens: {max_tokens_completion}, Prompt Tokens: {count_tokens(prompt)}") # Debug
                response = await async_openai_client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens_completion) # Use renamed param
                content = response.choices[0].message.content
                # print(f"[LLM Call Debug] Response Tokens: {count_tokens(content if content else '')}") # Debug
                return content.strip() if content else None
            except OpenAIError as e:
                print(f"[LLM Error] API call attempt {attempt+1}/{retry_attempts+1} failed ({model}): {type(e).__name__} - {e}")
                if "context_length_exceeded" in str(e):
                     print(f"[LLM Error] Context length exceeded for model {model}. Prompt may be too long.")
                     return None # Don't retry context length errors usually
                if attempt < retry_attempts:
                    wait_time = delay * (2 ** attempt); print(f"[LLM Retry] Retrying in {wait_time:.2f}s..."); await asyncio.sleep(wait_time)
                else: # Moved 'else' for max retries outside the 'if attempt < retry_attempts' block
                    print(f"[LLM Error] Max retries reached for {model}."); return None
            except Exception as e:
                print(f"[LLM Error] Unexpected error during API call ({model}): {type(e).__name__} - {e}"); traceback.print_exc(); return None
    # This return was potentially unreachable due to the 'else' block above, but kept for safety.
    return None


# --- Extraction ---
async def extract_details_from_chunk(
    chunk: str, url: str, model: str # Requires explicit model
) -> Optional[Dict[str, Any]]:
    """ Extracts details using the specified LLM model. """
    if not chunk: return None
    system_message = """You are an expert technical documentation analyst tasked with extracting specific, actionable details.
Analyze the provided text chunk meticulously.
PRIORITIZE the extraction and preservation of:
- Complete code examples (including imports, setup, and usage).
- Specific function/method names, their arguments, data types, and return values.
- Detailed step-by-step instructions for setup, configuration, or usage workflows.
- Configuration parameters, environment variables, or command-line arguments.
- Key definitions or explanations crucial for understanding implementation.

DO NOT over-summarize technical details.
Preserve the original phrasing for technical accuracy where possible.
Format code blocks clearly using ``` markers.
Extract ONLY the relevant technical information. Avoid introductions, conclusions, or explanations *about* the chunk itself.
If the chunk contains absolutely no specific technical details (e.g., only navigation links, headers, marketing text, or generic prose), respond with the single word: IRRELEVANT"""
    prompt = f"""Please analyze the following text chunk from the webpage {url} based on the instructions in the system message.\nText Chunk:\n\"\"\"\n{chunk}\n\"\"\"\n\nExtracted Details:"""
    tokenizer = get_tokenizer()
    chunk_tokens = count_tokens(chunk, tokenizer)
    max_completion_tokens = min(4096, max(600, int(chunk_tokens * 0.40)))

    response = await get_llm_response(prompt, system_message, model=model, temperature=0.05, max_tokens_completion=max_completion_tokens)
    if response:
        if response.strip().upper() == "IRRELEVANT": return None
        else:
            # Check for minimal substance
            if len(response) < 30:
                print(f"[Summarize] Warning: Chunk analysis from {url} seems too short, discarding. Content: '{response[:100]}...'")
                return None
            return {"source_url": url, "analysis": response}
    else: print(f"[Summarize] Failed to get analysis for chunk from {url}."); return None


# --- Consolidation ---
async def consolidate_summaries(
    input_texts: List[str], # Can be raw extractions or intermediate summaries
    topic: str,
    model: str, # Requires explicit model
    level: int = 0 # 0 for initial consolidation, 1+ for subsequent levels
) -> Optional[str]:
    """ Consolidates list of text details using the specified LLM model. Handles truncation if necessary. """
    if not input_texts: return None

    if level == 0:
        combined_input = f"# Raw Extracted Details for: {topic}\n\n---\n[Extracted Chunk Details]\n---\n\n"
        combined_input += "\n\n---\n[Extracted Chunk Details]\n---\n\n".join(input_texts)
        system_message = f"""You are a technical writer synthesizing extracted documentation details about '{topic}'.
Combine the following pieces of extracted information into a single, coherent, well-structured technical document optimized for a PLAIN TEXT (.txt) file.
Key Requirements:
- **Preserve Detail:** Retain specific code examples, step-by-step instructions, configuration details, and function/API signatures from the input. DO NOT over-summarize these specific elements.
- **Logical Flow:** Organize the information logically, grouping related concepts together.
- **Remove Redundancy (Carefully):** Eliminate clear duplication of *explanatory text*, but PRESERVE multiple code examples or instruction sets even if they illustrate similar points, as variations are often important.
- **Formatting:** Use paragraphs for explanations. Use simple lists (* or -) for steps or features. Enclose ALL code blocks, snippets, or commands in ``` markers. Use simple line breaks for structure. AVOID complex Markdown.
- **Accuracy:** Ensure the output is based *only* on the provided details.
- **Directness:** Avoid introductory or concluding phrases. Start directly with the technical content.
"""
    else:
        combined_input = f"# Intermediate Summaries for: {topic}\n\n---\n[Intermediate Summary]\n---\n\n"
        combined_input += "\n\n---\n[Intermediate Summary]\n---\n\n".join(input_texts)
        system_message = f"""You are a senior technical writer refining and synthesizing intermediate technical summaries about '{topic}'.
Combine the following summaries into a final, comprehensive, and well-structured technical document suitable for a PLAIN TEXT (.txt) file.
Key Requirements:
- **Merge & Enhance:** Integrate related concepts smoothly from the different summaries.
- **Preserve Critical Detail:** Ensure that distinct code examples, specific setup steps, configuration parameters, and detailed usage instructions from all intermediate summaries are retained in the final output. Do not drop these specifics.
- **Ensure Flow:** Create a clear narrative flow.
- **Eliminate Redundancy (Carefully):** Remove duplicated explanatory sentences, but KEEP variations in code examples or instructions.
- **Formatting:** Structure using paragraphs and simple lists (* or -). Enclose ALL code blocks in ``` markers. Use simple line breaks. AVOID complex Markdown.
- **Accuracy:** Base the output *only* on the provided intermediate summaries.
- **Directness:** Avoid introductions or conclusions. Start directly with the technical content.
"""

    tokenizer = get_tokenizer()
    total_input_tokens = count_tokens(combined_input, tokenizer)

    max_input_tokens = get_model_context_limit(model)
    completion_buffer = 4096 # Standard max completion for many models
    safe_input_token_limit = max_input_tokens - completion_buffer - 500 # Extra buffer

    print(f"[Consolidate L{level}] Preparing {total_input_tokens:,} tokens for consolidation using {model} (Limit: ~{safe_input_token_limit:,}).")

    if total_input_tokens > safe_input_token_limit:
        print(f"[Consolidate L{level}] Warning: Input ({total_input_tokens:,} tk) > safe limit ({safe_input_token_limit:,} tk). Truncating.")
        if tokenizer:
            encoded = tokenizer.encode(combined_input, disallowed_special=())
            truncated_encoded = encoded[:safe_input_token_limit]
            try:
                truncated_input = tokenizer.decode(truncated_encoded)
            except UnicodeDecodeError:
                 print("[Consolidate L{level}] Warning: Truncation caused decode error, reducing slightly.")
                 truncated_input = tokenizer.decode(truncated_encoded[:-10]) # Fallback
        else:
            estimated_chars = safe_input_token_limit * 4
            truncated_input = combined_input[:estimated_chars]
        combined_input = truncated_input
        total_input_tokens = count_tokens(combined_input, tokenizer)
        print(f"[Consolidate L{level}] Truncated input tokens: {total_input_tokens:,}")

    prompt = f"""Please synthesize the following information into a single, well-structured technical document about '{topic}', following the system instructions (prioritize detail retention, use plain text formatting).\n\nInformation:\n\"\"\"\n{combined_input}\n\"\"\"\n\nFinal Synthesized Documentation (Plain Text):"""

    max_completion_tokens = completion_buffer

    consolidated_text = await get_llm_response(
        prompt,
        system_message,
        model=model,
        temperature=0.1, # Low temp for factual consolidation
        max_tokens_completion=max_completion_tokens
    )

    if consolidated_text:
        print(f"[Consolidate L{level}] Consolidation successful ({len(consolidated_text)} chars).")
    else:
        print(f"[Consolidate L{level}] Consolidation failed.")
    return consolidated_text


# --- Rating ---
async def rate_output_quality(
    consolidated_text: str, topic: str, model: str = RATING_MODEL
) -> Optional[Dict[str, Any]]:
    """ Rates output quality using RATING_MODEL from config """
    if not consolidated_text: return None
    system_message = f"""You are an expert developer reviewing automatically generated technical documentation for '{topic}'.
Evaluate the following documentation based on:
1.  **Clarity & Coherence:** Is the text well-organized, easy to understand, and logically structured?
2.  **Completeness & Accuracy:** Does it seem to cover the key aspects based on typical documentation for such a topic? (You don't have the original source, evaluate based on plausibility and internal consistency). Is the information presented accurately?
3.  **Conciseness:** Is there unnecessary repetition or verbose language?
4.  **Formatting:** Is the formatting (paragraphs, lists, code blocks) clear and appropriate for plain text?
Provide your assessment EXACTLY in the following format, ensuring the rating score is present and numeric:
**Overall Quality Rating:** [Score from 1 to 10]
**Justification:**
[Brief explanation for the rating, highlighting strengths and weaknesses based on the criteria above]
**Suggestions for Improvement:**
[Specific, actionable suggestions on how the documentation could be improved, if any]"""

    max_rating_input_tokens = get_model_context_limit(model) - 1000 # Leave buffer for prompt+completion
    tokenizer = get_tokenizer()
    token_count = count_tokens(consolidated_text, tokenizer)

    if token_count > max_rating_input_tokens:
        print(f"[Rating] Warning: Consolidated text ({token_count:,} tk) too long for rating model ({model}), truncating to {max_rating_input_tokens:,}.")
        if tokenizer:
            encoded = tokenizer.encode(consolidated_text, disallowed_special=())
            try:
                truncated_text = tokenizer.decode(encoded[:max_rating_input_tokens])
            except UnicodeDecodeError:
                truncated_text = tokenizer.decode(encoded[:max_rating_input_tokens-10]) # Fallback
        else: # Fallback if no tokenizer
            truncated_text = consolidated_text[:max_rating_input_tokens * 4] # Rough estimate
    else:
        truncated_text = consolidated_text

    prompt = f"""Please review the following automatically generated documentation regarding '{topic}' based on the criteria outlined in the system message.\n\nDocumentation:\n\"\"\"\n{truncated_text}\n\"\"\"\n\nAssessment:"""

    response = await get_llm_response(prompt, system_message, model=model, temperature=0.3)
    if response:
        # Initialize variables with defaults BEFORE the try block
        rating_score = None
        justification = "Parsing failed."
        suggestions = "Parsing failed."
        raw_response = response # Keep raw response regardless

        try:
            # --- Improved Parsing Logic ---
            # 1. Parse Rating Score (more flexible regex)
            rating_match = re.search(r"Overall Quality Rating\s*[:\-]?\s*(\b(?:10|[1-9])\b)", response, re.IGNORECASE | re.MULTILINE)
            # Assign rating_score only if match is found
            if rating_match:
                 rating_score = int(rating_match.group(1))

            # 2. Split into sections based on headers (more robust than numbered list regex)
            justification_header = "Justification:"
            suggestions_header = "Suggestions for Improvement:"

            # Define placeholders in case split fails
            parsed_justification = "Justification section not found."
            parsed_suggestions = "Suggestions section not found."

            parts = response.split(justification_header)
            if len(parts) > 1:
                # Text after "Justification:" header
                just_and_sugg = parts[1]
                sub_parts = just_and_sugg.split(suggestions_header)
                # Assign parsed justification if found
                parsed_justification = sub_parts[0].strip()
                if len(sub_parts) > 1:
                    # Assign parsed suggestions if found
                    parsed_suggestions = sub_parts[1].strip()
            else:
                # Try finding suggestions even if justification failed to split
                 sugg_parts = response.split(suggestions_header)
                 if len(sugg_parts) > 1:
                     parsed_suggestions = sugg_parts[1].strip()

            # Clean up potential initial empty lines or list markers if needed
            # Assign back to the main variables AFTER successful parsing
            justification = re.sub(r"^\s*[-*]?\s*", "", parsed_justification)
            suggestions = re.sub(r"^\s*[-*]?\s*", "", parsed_suggestions)

        except Exception as parse_e:
            print(f"[Rating] Error parsing rating response: {parse_e}")
            # Fallback values are already set outside the try block
            # Optionally, add more specific error info to justification/suggestions
            justification = f"Parsing failed ({parse_e}). See raw output."
            suggestions = "Parsing failed. See raw output."

        # Now justification and suggestions will always have a value
        rating_score_str = str(rating_score) if rating_score is not None else 'N/A'
        print(f"[Rating] Quality Rating received (Score: {rating_score_str}).")
        return {"rating_score": rating_score, "rating_justification": justification, "rating_suggestions": suggestions, "raw_rating_output": raw_response}
    else:
        print("[Rating] Failed to get quality rating.")
        return None


# --- Embedding Generation ---
async def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> Optional[List[float]]:
     """ Generates embeddings using EMBEDDING_MODEL """
     if not async_openai_client: print("[Embedding Error] OpenAI client not initialized."); return None
     if not text: return None
     try:
          # Replace newlines which can affect some embedding models
          text = text.replace("\n", " ")
          # Adhere to OpenAI's recommended max token limit for embeddings
          max_emb_tokens = 8190 # Max for text-embedding-3-small/large, text-embedding-ada-002
          tokenizer = get_tokenizer()
          token_count = count_tokens(text, tokenizer)
          if token_count > max_emb_tokens:
               print(f"[Embedding] Warning: Input text ({token_count} tokens) longer than max ({max_emb_tokens}). Truncating.")
               if tokenizer:
                    encoded = tokenizer.encode(text, disallowed_special=())
                    text = tokenizer.decode(encoded[:max_emb_tokens])
               else: text = text[:max_emb_tokens*4] # Fallback if no tokenizer
          response = await async_openai_client.embeddings.create(input=[text], model=model)
          return response.data[0].embedding
     except OpenAIError as e: print(f"[Embedding Error] API call failed ({model}): {type(e).__name__} - {e}")
     except Exception as e: print(f"[Embedding Error] Unexpected error ({model}): {type(e).__name__} - {e}"); traceback.print_exc()
     return None

# --- Cosine Similarity ---
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """ Calculates cosine similarity """
    # Ensure inputs are valid lists of floats
    if not vec1 or not vec2 or not isinstance(vec1, list) or not isinstance(vec2, list):
        # print("[Similarity Warn] Invalid input vectors provided.")
        return 0.0
    try:
        vec1_arr = np.array(vec1, dtype=np.float32)
        vec2_arr = np.array(vec2, dtype=np.float32) # Use different variable names
        if vec1_arr.shape != vec2_arr.shape:
            print(f"[Similarity Error] Vector shapes mismatch: {vec1_arr.shape} vs {vec2_arr.shape}")
            return 0.0
        # Calculate norms, handle potential zero division
        norm_vec1 = np.linalg.norm(vec1_arr)
        norm_vec2 = np.linalg.norm(vec2_arr)
        if norm_vec1 == 0 or norm_vec2 == 0:
            # print("[Similarity Warn] One or both vectors have zero magnitude.")
            return 0.0
        # Calculate dot product and similarity
        dot_product = np.dot(vec1_arr, vec2_arr)
        similarity = dot_product / (norm_vec1 * norm_vec2)
        # Clip to handle potential floating point inaccuracies slightly outside [-1, 1]
        return float(np.clip(similarity, -1.0, 1.0))
    except ValueError as ve:
        print(f"[Similarity Error] Could not convert vectors to arrays: {ve}")
        return 0.0
    except Exception as e:
        print(f"[Similarity Error] Unexpected error: {type(e).__name__} - {e}")
        traceback.print_exc()
        return 0.0


# --- Helper for Cost Estimation and Model Selection ---
async def _get_user_confirmation_and_model(
    step_name: str,
    estimated_total_input_tokens: int,
    estimated_total_output_tokens: int,
    model_option_1_name: str = "gpt-4o-mini",
    model_option_2_name: str = "gpt-4o"
) -> Optional[str]:
    """
    Calculates estimated total costs for two models for a potentially multi-step process,
    prompts user for selection and confirmation. Returns chosen model name or None if cancelled.
    """
    try:
        price_opt1 = get_model_pricing(model_option_1_name)
        price_opt2 = get_model_pricing(model_option_2_name)

        cost_opt1 = (estimated_total_input_tokens / 1_000_000 * price_opt1['input']) + \
                    (estimated_total_output_tokens / 1_000_000 * price_opt1['output'])
        cost_opt2 = (estimated_total_input_tokens / 1_000_000 * price_opt2['input']) + \
                    (estimated_total_output_tokens / 1_000_000 * price_opt2['output'])

        print("-" * 40)
        print(f"[User Checkpoint] Action: {step_name}")
        print(f"  - Estimated TOTAL Input Tokens (across all calls): {estimated_total_input_tokens:,}")
        print(f"  - Estimated TOTAL Output Tokens (across all calls): {estimated_total_output_tokens:,}")
        print("-" * 20)
        print(f"  Estimated Cost ({step_name}):")
        print(f"    1. {model_option_1_name}: ${cost_opt1:.4f}")
        print(f"    2. {model_option_2_name}: ${cost_opt2:.4f}")
        print("-" * 40)
        print(f"NOTE: Token counts and costs for '{step_name}' are rough estimates,")
        print(f"      especially with hierarchical processing.")


        chosen_model = None
        while chosen_model is None:
            # Allow skipping confirmation if explicitly set in config (e.g., for automated runs)
            if hasattr(config, 'AUTO_CONFIRM_MODELS') and config.AUTO_CONFIRM_MODELS:
                 print("[Processor] Auto-confirming model selection as per config.")
                 # Default to option 1 if auto-confirming
                 chosen_model = model_option_1_name
                 print(f"[Processor] Auto-selected: {chosen_model}")
                 break # Exit the loop

            try:
                choice = input(f"Select model for {step_name} (1 or 2, or 'c' to cancel): ").strip().lower()
                if choice == '1': chosen_model = model_option_1_name
                elif choice == '2': chosen_model = model_option_2_name
                elif choice == 'c': print(f"[Processor] User cancelled operation at model selection for {step_name}."); return None
                else: print("Invalid choice. Please enter '1', '2', or 'c'.")
            except (EOFError, KeyboardInterrupt):
                print(f"\n[Processor] User cancelled operation during model selection for {step_name}.")
                return None


        print(f"[Processor] User confirmed. Proceeding with {step_name} using {chosen_model}.")
        print("-" * 40)
        return chosen_model # Return the selected model name

    except Exception as e:
        print(f"[Processor Error] Failed during cost estimation/confirmation for {step_name}: {type(e).__name__} - {e}")
        traceback.print_exc();
        print("[Processor] Could not estimate cost or get confirmation. Aborting step.");
        return None

# --- Main Processing Orchestration ---
async def process_scraped_data(
    scraped_data: Dict[str, Dict[str, Any]],
    topic: str
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Orchestrates processing (cleaning/passing through), extraction, hierarchical consolidation, and rating.
    """
    if not async_openai_client:
        print("[Processor Error] OpenAI client not available. Cannot process data.")
        return None, None

    print("[Processor] Starting content processing...")
    print(f"  - Config Defaults: Extract={EXTRACTION_MODEL_DEFAULT}, Consolidate={CONSOLIDATION_MODEL_DEFAULT}, Rate={RATING_MODEL}")
    print(f"  - Chunk Size (Tokens): {CHUNK_SIZE_TOKENS}")
    print(f"  - LLM Concurrency Limit: {LLM_CONCURRENCY_LIMIT}")

    tokenizer = get_tokenizer()
    if not tokenizer:
         print("[Processor Error] Tokenizer could not be initialized. Cannot proceed with token counting.")
         return None, None

    # --- Phase 0: Aggregate Content (Apply Cleaning Conditionally) ---
    print("[Processor] Phase 0: Aggregating content...")
    aggregated_text = ""
    content_to_process: Dict[str, str] = {} # Store text (cleaned HTML or raw Markdown) per URL
    processed_page_count = 0
    skipped_no_content = 0
    skipped_empty_after_processing = 0
    total_pages_input = len(scraped_data)

    for url, data in scraped_data.items():
        # *** MODIFICATION START ***
        original_content = data.get("content") # Use "content" key
        content_type = data.get("type", "html") # Get type, default to "html"

        if not original_content:
            skipped_no_content += 1
            continue

        processed_text: Optional[str] = None
        if content_type == "markdown":
            processed_text = original_content # Use raw Markdown directly
            # print(f"[Processor Debug] Using raw Markdown for {url}") # Optional debug
        elif content_type == "html":
            processed_text = clean_html_content(original_content) # Clean HTML
            # print(f"[Processor Debug] Cleaned HTML for {url}") # Optional debug
        else:
            print(f"[Processor Warning] Unknown content type '{content_type}' for {url}. Skipping.")
            skipped_empty_after_processing +=1
            continue

        if not processed_text: # Check if processed_text is empty after potential cleaning
            skipped_empty_after_processing += 1
            continue
        # *** MODIFICATION END ***

        # Store the processed text (either cleaned HTML or raw Markdown)
        content_to_process[url] = processed_text
        processed_page_count += 1
        # Aggregate for token estimation (use processed text)
        aggregated_text += processed_text + "\n\n" # Separator for token count

    if not content_to_process: # Check if any content remains *after* conditional processing
        print("[Processor] No processable content found after initial processing/cleaning. Exiting.")
        return None, None

    print(f"[Processor] Pages aggregated: {processed_page_count}/{total_pages_input} (Skipped: {skipped_no_content} no content, {skipped_empty_after_processing} empty after processing).")

    # --- STOP 1: Extraction Cost Estimation & Model Selection --- #
    extraction_input_tokens = count_tokens(aggregated_text, tokenizer)
    # Estimate extraction output tokens (e.g., 35% of input?) - Adjust estimate based on prompt
    est_extraction_output_tokens = int(extraction_input_tokens * 0.35)

    # Get default models from config for user selection prompt
    default_extract_model = EXTRACTION_MODEL_DEFAULT
    # Offer gpt-4o as the alternative if default is mini, or mini if default is gpt-4o/other
    alternative_extract_model = "gpt-4o" if "mini" in default_extract_model else "gpt-4o-mini"

    chosen_extraction_model = await _get_user_confirmation_and_model(
        step_name="Extraction",
        estimated_total_input_tokens=extraction_input_tokens,
        estimated_total_output_tokens=est_extraction_output_tokens,
        model_option_1_name=default_extract_model,
        model_option_2_name=alternative_extract_model
    )

    if not chosen_extraction_model: return None, None # User cancelled

    # --- Phase 1: Chunk and Launch Extraction Tasks --- #
    print(f"[Processor] Phase 1: Chunking and Launching Extraction Tasks using {chosen_extraction_model}...")
    extraction_tasks = []
    total_chunks = 0
    # Use the 'content_to_process' dict which holds the correctly processed text
    for url, text_to_chunk in content_to_process.items():
        chunks = chunk_text(text_to_chunk, max_tokens=CHUNK_SIZE_TOKENS)
        total_chunks += len(chunks)
        for i, chunk in enumerate(chunks):
            task = asyncio.create_task(
                extract_details_from_chunk(chunk, url, model=chosen_extraction_model) # Pass chosen model
            )
            extraction_tasks.append(task)
    print(f"[Processor] Launched {len(extraction_tasks)} extraction tasks for {total_chunks} chunks across {len(content_to_process)} pages.")

    # --- Phase 2: Gather Extraction Results ---
    print("[Processor] Phase 2: Gathering extraction results...")
    all_extracted_details_dicts: List[Dict[str, Any]] = []
    successful_extractions = 0
    failed_extractions = 0
    irrelevant_or_failed = 0

    if extraction_tasks:
        extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        for result in extraction_results:
            if isinstance(result, Exception):
                print(f"[Processor Error] Extraction task failed: {type(result).__name__} - {result}")
                failed_extractions += 1
                irrelevant_or_failed += 1
            elif result is None: irrelevant_or_failed += 1 # Handled None/IRRELEVANT
            elif isinstance(result, dict) and result.get("analysis"): # Ensure it's a dict with analysis
                all_extracted_details_dicts.append(result)
                successful_extractions += 1
            else:
                 print(f"[Processor Warning] Unexpected extraction result type or format: {type(result)}")
                 irrelevant_or_failed += 1 # Count unexpected format as failed/irrelevant
        print(f"[Processor] Successfully extracted details from {successful_extractions} chunks (Failures/Irrelevant: {irrelevant_or_failed}).")
    else:
        print("[Processor] No extraction tasks were launched.")
        return None, None

    if not all_extracted_details_dicts:
         print("[Processor] No details were successfully extracted. Cannot proceed to consolidation.")
         return None, None

    # Extract just the analysis strings for consolidation input
    all_extracted_details = [d['analysis'] for d in all_extracted_details_dicts if d.get('analysis')]
    consolidation_input_tokens = sum(count_tokens(t, tokenizer) for t in all_extracted_details)

    # --- STOP 2: Consolidation Cost Estimation & Model Selection ---
    # Estimate output based on total extracted tokens
    est_intermediate_output = int(consolidation_input_tokens * 0.40) # Adjusted estimate
    est_final_output = int(est_intermediate_output * 0.85) # Adjusted estimate
    est_consolidation_output_tokens = est_intermediate_output + est_final_output

    # Get default models from config for user selection prompt
    default_consolidate_model = CONSOLIDATION_MODEL_DEFAULT
    alternative_consolidate_model = "gpt-4o" if "mini" in default_consolidate_model else "gpt-4o-mini"

    chosen_consolidation_model = await _get_user_confirmation_and_model(
        step_name="Consolidation (Hierarchical)",
        estimated_total_input_tokens=consolidation_input_tokens, # Total input to first level
        estimated_total_output_tokens=est_consolidation_output_tokens, # Rough total output estimate
        model_option_1_name=default_consolidate_model,
        model_option_2_name=alternative_consolidate_model
    )

    if not chosen_consolidation_model: return None, None # User cancelled

    # --- Phase 3: Hierarchical Consolidation ---
    print(f"[Processor] Phase 3: Consolidating {consolidation_input_tokens:,} extracted tokens hierarchically using {chosen_consolidation_model}...")

    current_texts_to_consolidate = all_extracted_details
    consolidation_level = 0
    max_consolidation_levels = 5 # Safety break
    consolidated_output: Optional[str] = None

    consolidation_token_limit = get_model_context_limit(chosen_consolidation_model)
    # Use a slightly safer margin for input tokens in consolidation
    safe_consolidation_input_limit = int(consolidation_token_limit * 0.80)

    while consolidation_level < max_consolidation_levels:
        current_total_tokens = sum(count_tokens(t, tokenizer) for t in current_texts_to_consolidate)
        print(f"[Consolidate L{consolidation_level}] Input level {consolidation_level} with {len(current_texts_to_consolidate)} text(s), total tokens: {current_total_tokens:,}")

        # If only one text remains, or the total fits in one call, perform final consolidation
        if len(current_texts_to_consolidate) == 1 or current_total_tokens <= safe_consolidation_input_limit:
            if len(current_texts_to_consolidate) > 1:
                print(f"[Consolidate L{consolidation_level}] Final consolidation: Combining {len(current_texts_to_consolidate)} texts ({current_total_tokens:,} tokens).")
                consolidated_output = await consolidate_summaries(
                     current_texts_to_consolidate, topic, chosen_consolidation_model, level=consolidation_level
                )
            else:
                 # If only one text left, it's the final output
                 print(f"[Consolidate L{consolidation_level}] Final consolidation: Single text remaining.")
                 consolidated_output = current_texts_to_consolidate[0]
            break # Exit the loop

        # --- Batching Logic ---
        print(f"[Consolidate L{consolidation_level}] Batching {len(current_texts_to_consolidate)} texts for intermediate consolidation...")
        batches: List[List[str]] = []
        current_batch: List[str] = []
        current_batch_tokens = 0

        for text in current_texts_to_consolidate:
            text_tokens = count_tokens(text, tokenizer)
            # Check if adding this text exceeds the limit for a batch
            if current_batch_tokens + text_tokens <= safe_consolidation_input_limit:
                current_batch.append(text)
                current_batch_tokens += text_tokens
            else:
                # Finalize the current batch if it has content
                if current_batch:
                    batches.append(current_batch)
                # Start a new batch with the current text
                # Handle case where single text > limit
                if text_tokens > safe_consolidation_input_limit:
                     print(f"[Consolidate L{consolidation_level}] Warning: Single text ({text_tokens} tk) > batch limit ({safe_consolidation_input_limit} tk). Adding as own batch (will be truncated).")
                     batches.append([text]) # Add as its own batch, will likely be truncated by consolidate_summaries
                     current_batch = []
                     current_batch_tokens = 0
                else:
                    current_batch = [text]
                    current_batch_tokens = text_tokens

        # Add the last batch if it exists
        if current_batch: batches.append(current_batch)

        print(f"[Consolidate L{consolidation_level}] Created {len(batches)} batches.")

        # --- Run Consolidation on Batches ---
        consolidation_tasks = []
        for batch in batches:
             task = asyncio.create_task(
                 consolidate_summaries(batch, topic, chosen_consolidation_model, level=consolidation_level)
             )
             consolidation_tasks.append(task)

        intermediate_results = await asyncio.gather(*consolidation_tasks, return_exceptions=True)

        # --- Prepare for Next Level --- #
        next_level_texts = []
        errors_this_level = 0
        for result in intermediate_results:
             if isinstance(result, Exception):
                 print(f"[Consolidate L{consolidation_level} Error] Consolidation task failed: {type(result).__name__} - {result}")
                 errors_this_level += 1
             elif result: # Only add non-empty results
                 next_level_texts.append(result)

        if not next_level_texts:
             print(f"[Consolidate L{consolidation_level}] Error: No successful intermediate summaries generated. Aborting consolidation.")
             consolidated_output = None # Ensure output is None
             break # Exit loop

        print(f"[Consolidate L{consolidation_level}] Completed level {consolidation_level}. Generated {len(next_level_texts)} intermediate texts. Errors: {errors_this_level}.")
        current_texts_to_consolidate = next_level_texts
        consolidation_level += 1

        if consolidation_level >= max_consolidation_levels:
             print(f"[Consolidate Error] Reached max consolidation levels ({max_consolidation_levels}). Aborting.")
             consolidated_output = None
             break # Exit loop


    if not consolidated_output:
        print("[Processor] Consolidation failed or produced no output.")
        return None, None

    # --- Phase 4: Rate Quality ---
    print(f"[Processor] Phase 4: Rating final output quality using {RATING_MODEL}...")
    quality_rating = await rate_output_quality(
        consolidated_output, topic, model=RATING_MODEL # Uses config model
    )

    print("[Processor] Content processing finished.")
    return consolidated_output, quality_rating

# --- Standalone Execution ---
if __name__ == "__main__":
    print("This script `content_processor.py` is intended to be imported as a module.")