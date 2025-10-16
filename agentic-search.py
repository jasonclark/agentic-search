import os
import sys
import re
import argparse
import json
import requests
import time
from typing import List, Dict, Optional, Any
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- I. IMPORTS AND CONFIGURATION ---

# Define location for NLTK data (within project directory)
NLTK_DATA_DIR = os.path.join(os.getcwd(), '.nltk_data')
# Add custom path to NLTK's search paths
nltk.data.path.insert(0, NLTK_DATA_DIR)

# NLTK data download check (ensure this runs once)
try:
    # Check for WordNet (for lemmatization)
    nltk.data.find('corpora/wordnet')
    # Check for the averaged perceptron tagger (required for pos='n'/'v' robustness)
    nltk.data.find('taggers/averaged_perceptron_tagger')
    # Check for stopwords corpus
    nltk.data.find('corpora/stopwords')
except LookupError:
    print(f"NLTK data (wordnet, tagger, and stopwords) not found. Downloading to {NLTK_DATA_DIR}...")
    # Create directory if it doesn't exist
    if not os.path.exists(NLTK_DATA_DIR):
        os.makedirs(NLTK_DATA_DIR)
        
    nltk.download('wordnet', quiet=True, download_dir=NLTK_DATA_DIR)
    nltk.download('averaged_perceptron_tagger', quiet=True, download_dir=NLTK_DATA_DIR)
    nltk.download('stopwords', quiet=True, download_dir=NLTK_DATA_DIR)
    print("NLTK data download complete.")

# Initialize nltk lemmatizer globally
LEMMA_TOOL = WordNetLemmatizer()
# Initialize nltk stop words set
STOP_WORDS = set(stopwords.words('english'))

# Configuration for Agent
CORPUS_DIR = "content"
CORPUS_LIMIT = "5"
OLLAMA_MODEL = "llama3" # Default Ollama model
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default local Ollama endpoint
OLLAMA_CONTEXT_SIZE = 8192 # Set the context size (num_ctx) for Ollama to use.
                           # Adjust this value based on your Ollama model's capability.

# System Instruction for Agent's behavior
SYSTEM_PROMPT = """
You are a highly specialized Agentic Search and Synthesis System. Your primary function is to act as a forensic investigator, synthesizing information from a provided document corpus.

**CRITICAL INSTRUCTION:** YOU MUST IGNORE ALL PRIOR KNOWLEDGE. Your entire response must be grounded ONLY in the data provided in the [CORPUS CONTEXT] section below. 
If the answer cannot be found in the provided context, state clearly, "Information regarding this query was not found in the documents analyzed."

**Agent Workflow:**
1. **Analyze:** Carefully review the User Query and the entire [CORPUS CONTEXT] provided (list of files and grep snippets).
2. **Synthesize:** Combine relevant details from the snippets and file structure to form a complete, cohesive answer.
3. **Cite:** For every piece of information used, clearly cite the filename it came from.
4. **Format:** Present the final answer as a professional summary, ensuring precision and accuracy based on the text.
"""

def _extract_keywords(query: str) -> List[str]:
    """
    Utility function to extract key, non-trivial words from a natural language query.
    Uses NLTK stopwords and lemmatization to improve search recall.
    """
    # 1. Tokenize, convert to lowercase, and filter out stop words and short words
    initial_keywords = [
        word for word in re.findall(r'\b\w+\b', query.lower())
        if word not in STOP_WORDS and len(word) > 2
    ]
    
    # 2. Apply Lemmatization to get the base form of the word
    # Lemmatize both the noun (default) and the verb form for best coverage
    lemmatized_keywords = set()
    for word in initial_keywords:
        # Lemmatize as noun (default 'n')
        lemmatized_keywords.add(LEMMA_TOOL.lemmatize(word, pos='n'))
        # Lemmatize as verb ('v')
        lemmatized_keywords.add(LEMMA_TOOL.lemmatize(word, pos='v'))
        
    # Include the original keyword list just in case lemmatization fails (e.g., proper nouns)
    final_keywords = lemmatized_keywords.union(set(initial_keywords))
    
    return list(final_keywords) # Return unique and lemmatized keywords

# --- II. AGENT'S TOOLS (Glob and Grep) ---

def glob_files(corpus_dir: str, query: str, corpus_limit: int) -> List[str]:
    """
    Agent's file discovery tool 
    Returns a strictly selective list of file paths.
    
    Strict Selectivity Logic:
    1. Filter: Find ALL files where query keywords appear in the filename.
    2. Sort: Rank matching files by recency (newest first).
    3. Limit: Return the top N matching files (up to CORPUS_LIMIT).
    4. Fallback: If ZERO files match the keywords, return the top N most recent files.
    """
    all_files = []
    keywords = _extract_keywords(query)
    
    print(f"Agent's Glob Tool: Using keywords for filename search: {keywords}")

    try:
        if not os.path.exists(corpus_dir):
            os.makedirs(corpus_dir)
            print(f"Created corpus directory: {corpus_dir}")
            
        for root, _, files in os.walk(corpus_dir):
            for file_name in files:
                # Skip hidden/system files (starting with '.')
                if file_name.startswith('.'):
                    print(f"Skipping hidden/system file: {file_name}")
                    continue
                
                file_path = os.path.join(root, file_name)
                if not os.path.isfile(file_path) or not os.access(file_path, os.R_OK):
                    continue
                    
                stat_info = os.stat(file_path)
                
                # Check for keyword match in the filename
                is_keyword_match = any(k in file_name.lower() for k in keywords)
                
                all_files.append({
                    'path': file_path,
                    'mtime': stat_info.st_mtime,
                    'is_keyword_match': is_keyword_match
                })
        
        # 1. Separate matches from non-matches
        keyword_matches = [f for f in all_files if f['is_keyword_match']]
        other_files = [f for f in all_files if not f['is_keyword_match']]
        
        # 2. Sort both lists by recency (newest first)
        keyword_matches.sort(key=lambda x: x['mtime'], reverse=True)
        other_files.sort(key=lambda x: x['mtime'], reverse=True)
        
        selected_files = []
        
        if keyword_matches:
            # PRIMARY ACTION: Return only the keyword matches, limited by corpus_limit
            selected_files = keyword_matches[:int(CORPUS_LIMIT)]
            print("Progress Signal: Prioritizing strict keyword matches.")
        else:
            # FALLBACK ACTION: If no keywords match, return the most recent files.
            selected_files = other_files[:int(CORPUS_LIMIT)]
            print("Progress Signal: No keyword matches found in filenames. Falling back to most recent files.")

        file_paths = [f['path'] for f in selected_files] 
        print(f"Progress Signal: Glob selected {len(file_paths)} files (Max: {corpus_limit}).")

        return file_paths
        # --------------------------------

    except FileNotFoundError:
        print(f"Error: Corpus directory '{corpus_dir}' not found. Please create it and add documents.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during glob: {e}")
        return []

def grep_files(file_paths: List[str], query: str, context_lines: int = 1) -> Dict[str, List[str]]:
    """
    Agent's search tool performs fuzzy keyword search across content of specified files.
    Returns dictionary mapping file path to a list of matching context snippets.
    """
    results: Dict[str, List[str]] = {}
    keywords = _extract_keywords(query)
    keyword_patterns = [re.escape(k) for k in keywords]
    
    if not keyword_patterns:
        print("Agent's Grep Tool: No effective keywords extracted for searching.")
        return {}

    # Compile a regex pattern to find any of the keywords
    # Simulates fast, exact-match searching
    combined_pattern = re.compile(f'({"|".join(keyword_patterns)})', re.IGNORECASE)
    
    print(f"Agent's Grep Tool: Searching for keyword patterns: {', '.join(keywords)} in {len(file_paths)} files.")

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Split content into lines for line-by-line processing
            lines = content.splitlines()
            file_matches = []
            
            for i, line in enumerate(lines):
                if combined_pattern.search(line):
                    # Found a match. Extract the context (line itself)
                    # Highlight the keywords in the line for the LLM's attention
                    highlighted_line = combined_pattern.sub(lambda m: f'**{m.group(0)}**', line)
                    
                    snippet = f"Line {i+1}: {highlighted_line.strip()}"
                    file_matches.append(snippet)

            if file_matches:
                # Limit the number of snippets to prevent flooding the LLM, 
                # relying on Agent (the LLM itself) to ask for more if needed.
                # 10 is a reasonable limit for initial context.
                results[file_path] = file_matches[:10] 

        except UnicodeDecodeError:
            # Handles binary files (like PDFs) that we can't easily read as text.
            pass # Silently skip non-text files for grep
        except Exception as e:
            print(f"Could not grep file {file_path}: {e}")

    return results

# --- III. THE LLM'S REASONING FUNCTION ---

def query_ollama_agent(
    query: str, 
    context_files: List[str], 
    grep_results: Dict[str, List[str]], 
    model_name: str, 
    api_url: str,
    context_size: int
) -> str:
    """
    Agent's reasoning function. It takes the query and tool results,
    constructs a grounded prompt, and calls the Ollama API.
    """
    
    # 1. Format the Grep Results for the LLM's Context Window
    context_parts = []
    
    if context_files:
        context_parts.append("\n--- FILE LIST FOR REFERENCE ---")
        context_parts.append("Files considered in this investigation:\n" + "\n".join(f"- {path}" for path in context_files))
    
    if grep_results:
        context_parts.append("\n--- GREP SNIPPETS (CRITICAL CONTEXT) ---")
        for file_path, snippets in grep_results.items():
            context_parts.append(f"\nSource File: {file_path}")
            for snippet in snippets:
                context_parts.append(f"  > {snippet}")
        context_parts.append("--- END GREP SNIPPETS ---")
    else:
        context_parts.append("\n--- GREP SNIPPETS (CRITICAL CONTEXT) ---\nNo specific keywords were found in the analyzed documents.\n--- END GREP SNIPPETS ---")

    corpus_context = "\n".join(context_parts)

    # 2. Construct Final Prompt Payload
    
    # Final user prompt structure guides agent to use provided context immediately
    user_prompt = f"""
    [CORPUS CONTEXT]
    {corpus_context}
    
    [USER QUERY]
    Based SOLELY on the information above, please answer the following question:
    "{query}"
    """
    
    payload = {
        "model": model_name,
        "prompt": user_prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            # Low temperature encourages factual, grounded responses.
            "temperature": 0.2,
            "num_ctx": context_size # Pass the user-defined context size
        }
    }

    # 3. Call Ollama API with retry logic (Exponential Backoff)
    max_retries = 3
    delay = 1
    
    print(f"\nAgent Reasoning: Sending request to Ollama ({model_name})...")

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, json=payload, timeout=120)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            # Extract the content from response
            return result.get("response", "Error: No response text found in Ollama output.")

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                # Retry on connection error or temporary server error
                print(f"Ollama API request failed (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                # Final failure
                error_message = f"Critical Error: Ollama API call failed after {max_retries} attempts. Ensure Ollama is running at {api_url} and the model '{model_name}' is available. Details: {e}"
                return error_message
    
    return "Unknown error during Ollama communication." # Fallback

# --- IV. EXECUTION SETUP ---

def main():
    """
    The main execution function that orchestrates the agent's workflow.
    """
    # 1. Argument Parsing and Input Check
    parser = argparse.ArgumentParser(
        description="Agentic Search System: Queries a document corpus using LLM-driven investigation (Glob/Grep tools)."
    )
    parser.add_argument(
        "query", 
        nargs='?', 
        default=None, 
        help="The search query or question to investigate in the corpus."
    )
    args = parser.parse_args()

    # If script is run without arguments, ask user interactively
    query = args.query
    if not query:
        print("Agent Initializing. Please enter the query you want the agent to investigate.")
        user_input_query = input("Query: ")
        if not user_input_query.strip():
            print("Query cannot be empty. Exiting.")
            sys.exit(1)
        query = user_input_query.strip()

    print("\n" + "="*50)
    print(f"| Starting Agentic Search Investigation: ")
    print(f"| Query: {query}")
    print(f"| Model: {OLLAMA_MODEL}")
    print("="*50 + "\n")

    # 2. Agent Orchestration

    # Step A: Glob (File Discovery)
    print("Agent Step 1/3 (Thinking...): Running Glob Tool to identify relevant files...")
    file_paths = glob_files(CORPUS_DIR, query, CORPUS_LIMIT) 
    
    if not file_paths:
        print("Progress Signal: No documents found in the corpus. Skipping analysis.")
        final_answer = "Error: The corpus directory is empty or inaccessible. Please populate the 'content' folder with documents."
    else:
        print(f"Progress Signal: Glob identified {len(file_paths)} potential files for analysis.")
        
        # Step B: Grep (Context Extraction)
        print("Agent Step 2/3 (Thinking...): Running Grep Tool to extract critical context snippets...")
        grep_results = grep_files(file_paths, query)
        
        if not grep_results:
            print("Progress Signal: Grep found no keyword matches in the selected files.")
            # If grep finds nothing, still call the LLM to get final "not found" response
            final_answer = query_ollama_agent(
                query, 
                file_paths, 
                grep_results, 
                OLLAMA_MODEL, 
                OLLAMA_API_URL,
                OLLAMA_CONTEXT_SIZE 
            )
        else:
            total_snippets = sum(len(snippets) for snippets in grep_results.values())
            print(f"Progress Signal: Grep successfully extracted {total_snippets} relevant snippets.")
            # DEBUGGING STEP: Print raw context to see snippets passed to LLM
            print("\n*** DEBUG: GREP RAW CONTEXT EXTRACTED ***")
            for file_path, snippets in grep_results.items():
                print(f"--- File: {file_path} ---")
                for snippet in snippets:
                    print(snippet) # Print the entire multi-line snippet
            print("*****************************************\n")

            # Step C: Reasoning (Synthesis)
            print("Agent Step 3/3 (Thinking...): Running LLM Reasoning Engine for final synthesis...")
            final_answer = query_ollama_agent(
                query, 
                file_paths, 
                grep_results, 
                OLLAMA_MODEL, 
                OLLAMA_API_URL,
                OLLAMA_CONTEXT_SIZE
            )

    # 3. Output
    print("\n" + "="*50)
    print("--- AGENT CONCLUSION ---")
    print("="*50)
    print(final_answer)
    print("\n--- Investigation Complete ---")

if __name__ == "__main__":
    main()