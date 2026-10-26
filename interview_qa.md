# Interview Questions & Answers

## 1. Architecture & Design

**Q: Why did you choose a multi-agent architecture (LangGraph) over a single LLM call?**
A: A single LLM call struggles with complex reasoning over multiple data modalities (structured CSVs and unstructured PDFs). By splitting the task into specialized agents (Planner, Retriever, Analyst, Critic), the system becomes more explainable, easier to debug, and less prone to hallucination. The Planner extracts entities, the Analyst focuses on synthesizing data, and the Critic acts as a guardrail to ensure the final report is grounded in evidence.

**Q: Why did you use FAISS and `sentence-transformers` instead of a hosted vector database like Pinecone or OpenAI embeddings?**
A: This project was designed to run entirely locally without requiring expensive GPUs or cloud APIs. `sentence-transformers/all-MiniLM-L6-v2` is highly efficient and lightweight for generating embeddings, and FAISS (Facebook AI Similarity Search) provides excellent in-memory vector search capabilities that run well on standard CPUs.

## 2. RAG (Retrieval-Augmented Generation)

**Q: What chunking strategy did you use and why?**
A: I used the `RecursiveCharacterTextSplitter` with a chunk size of 500 and an overlap of 50 characters. This strategy recursively tries to split text at natural boundaries (paragraphs, then sentences, then words) to preserve context. The overlap ensures that concepts split across two chunks maintain their relationship, improving retrieval accuracy for policy clauses.

**Q: How would you improve the retrieval phase if the policies were much larger and more complex?**
A: I would implement advanced RAG techniques such as:
1.  **Parent-Child Chunking (Auto-merging Retriever):** Retrieve small, highly specific chunks but pass the larger parent document context to the LLM.
2.  **Hybrid Search:** Combine dense vector search (FAISS) with sparse keyword search (BM25) to handle exact keyword matches (e.g., specific policy codes).
3.  **Re-ranking:** Use a Cross-Encoder to re-rank the top K results retrieved by the vector database for higher precision.

## 3. Data Handling

**Q: How does the system handle structured data versus unstructured data?**
A: Unstructured data (policies) is embedded and stored in FAISS for semantic similarity search. Structured data (transactions, claims, merchants) is handled via Python tools wrapped around Pandas DataFrames. The Planner agent extracts the specific `transaction_id`, which the Analysis agent then uses to directly query the Pandas DataFrame (via tool invocation) for exact, hallucination-free facts.

## 4. Evaluation & Productionization

**Q: How do you evaluate the performance of this system?**
A: For this prototype, I established a custom evaluation framework focusing on:
1.  **Retrieval Relevance:** Does FAISS return the correct policy chunk for a given query?
2.  **Groundedness:** Does the Critic agent's report contain facts that exist *only* in the provided context (no hallucination)?
In a production setting, I would use a framework like **RAGAS** to systematically measure faithfulness, answer relevance, and context precision against a golden dataset.

**Q: How would you scale this to production?**
A: 
1.  Move from FAISS to a distributed vector database like Milvus or Qdrant.
2.  Replace Pandas with a robust SQL/NoSQL database (e.g., PostgreSQL).
3.  Deploy the LangGraph workflow as a FastAPI microservice and use Celery/Redis for asynchronous task queues, especially for the slow LLM inference steps.
4.  Upgrade the local LLM to a hosted, more powerful model (e.g., GPT-4o or Claude 3.5 Sonnet) if data privacy allows, or use vLLM for faster local inference on dedicated GPU instances.
