# Agentic Search Investigator

This repository contains a Python-based search agent built on the principles of **Agentic Search**. Instead of relying on the RAG pipeline (chunking, embeddings, reranking), this agent uses an intelligent LLM (via Ollama) and tools (`Glob` and `Grep` simulation) to investigate a document corpus directly.

## 💡 Concept: Why Agentic Search?

Traditional Retrieval-Augmented Generation (RAG) was a workaround for LLMs with small context windows. As context windows grow, the need for vector databases, complex chunking, and similarity search diminishes.

This agent operates like a forensic investigator:

1.  **Glob (File Discovery):** Quickly scans filenames to identify the most likely relevant files.
    
2.  **Grep (Exact Context):** Uses fast, exact-match keyword searching (lemmatized and stemmed for high recall) to extract only the most pertinent lines.
    
3.  **LLM Reasoning:** Synthesizes the exact context fragments provided by the tools, ignoring all prior knowledge, to generate a grounded, citable answer.
    

## 🚀 Getting Started

### Prerequisites

1.  **Python 3.8+**
    
2.  **Ollama:** The Ollama server must be installed and running locally on `http://localhost:11434`.
    
3.  **Model:** Pull the required model (default is `llama3`):
    
    ```
    ollama pull llama3
    
    ```
    

### Installation

Clone the repository and install the dependencies from `requirements.txt`.

1.  **Clone the repo and install:**
    
    ```
    git clone [YOUR_REPO_URL]
    cd [YOUR_REPO_NAME]
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    
    ```
    
2.  **NLTK Data Setup:** The script uses `nltk` for robust keyword extraction. Run the script once, and it will automatically download the necessary language files into the local `.nltk_data` folder for persistence.
    

### Corpus Setup

Create a folder named `content` in the root directory and place your `.txt`, `.md`, or other text-readable documents inside.

```
/your-repo
├── agentic_search.py
└── content/
    ├── annual_report_2024.txt
    └── project_overview.txt

```

## 💻 Usage

Run the script from your terminal, passing your query as an argument.

### Interactive Mode

If no query is provided, the script will prompt you:

```
python agentic_search.py

```

_(The script will then ask for your query input.)_

### Command-Line Mode

Provide the query directly in quotes:

```
python agentic_search.py "What are the core findings on revenue recognition policies from the latest report?"

```

## ⚙️ Core Agent Workflow

The `agentic_search.py` script orchestrates the following process:

1.  **Keyword Extraction:** The query is converted into a robust set of keywords using **NLTK Lemmatization** and official **Stop Word** filtering to maximize search recall.
    
2.  **Glob Selectivity:**
    
    -   Scans the `content/` folder.
        
    -   **Strictly prioritizes** files whose names contain the extracted keywords.
        
    -   Ranks the matches by **recency** (`mtime`).
        
    -   **Limits** the investigation to the top `CORPUS_LIMIT` (default 5) most relevant files.
        
3.  **Grep Extraction:**
    
    -   Searches the content of the selected files using a fast, combined regular expression matching the extracted keywords.
        
    -   Highlights the keywords in the resulting snippets (`**keyword**`) for the LLM's attention.
        
    -   Limits context to the top 10 matching snippets per file.
        
4.  **LLM Synthesis:** The agent is given a strict system prompt to **ignore prior knowledge** and synthesize the final answer based **only** on the provided file names and Grep snippets, including citations for every fact.