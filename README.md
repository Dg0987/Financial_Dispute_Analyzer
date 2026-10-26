# Agentic Financial Dispute Intelligence Engine

An end-to-end Agentic Retrieval-Augmented Generation (RAG) system built to investigate financial dispute cases like chargebacks, fraud, and refund requests.

Designed to run **100% locally** using Ollama, LangGraph, and FAISS.

## Features

- **Multi-Agent Workflow**: Utilizes a Planner, Retriever, Analyst, and Critic to synthesize information and generate evidence-backed reports.
- **Hybrid Data Processing**: Combines structured Pandas queries (transactions, claims) with unstructured vector search (policy PDFs).
- **Streamlit Interface**: Interactive UI to upload policies, ask investigation questions, and inspect agent reasoning.
- **Local & Lightweight**: Uses `all-MiniLM-L6-v2` for embeddings, FAISS for vector search, and Ollama for LLM inference. No cloud APIs required.

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally.
- A local LLM pulled via Ollama. By default, the code uses `llama3`.
  ```bash
  ollama pull llama3
  ```

### 2. Installation

1. Clone the repository and navigate into it.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Generate Synthetic Data

Generate the `transactions.csv`, `claims.csv`, `merchants.csv`, and synthetic policy PDFs:

```bash
python src/data_generation.py
```

### 4. Run the Application

Start the Streamlit frontend:

```bash
streamlit run app.py
```

## Project Structure

- `data/`: Contains synthetic structured CSVs and unstructured PDFs.
- `src/rag_pipeline.py`: FAISS vector store and embedding logic.
- `src/tools.py`: Pandas-based tools for agents to query structured data.
- `src/graph.py`: LangGraph state machine defining the multi-agent workflow.
- `app.py`: Streamlit frontend application.
- `architecture.md`: Detailed system design.
- `interview_qa.md`: Common interview questions regarding this project.

## How it Works

1. **Planner**: Parses the user query (e.g., "Investigate TXN00100") and extracts entities.
2. **Retriever**: Embeds the query and fetches relevant policy chunks from FAISS.
3. **Analyst**: Uses tools to fetch the exact transaction details from CSVs, combines it with the retrieved policies, and drafts an analysis.
4. **Critic**: Reviews the draft for hallucination and formats a final Markdown report with a confidence score.
