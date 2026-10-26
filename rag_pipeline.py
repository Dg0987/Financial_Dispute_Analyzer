import os
import pickle
from typing import List

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

POLICY_DIR = "data/policies"
KNOWLEDGE_BASE_DIR = "knowledge_base_cleaned"
FAISS_INDEX_PATH = "data/faiss_index"
BM25_INDEX_PATH = "data/bm25_index.pkl"

class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.bm25_retriever = None
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def load_documents(self) -> List[Document]:
        """Loads all PDFs from the POLICY_DIR and txt files from KNOWLEDGE_BASE_DIR."""
        documents = []
        if os.path.exists(POLICY_DIR):
            for filename in os.listdir(POLICY_DIR):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(POLICY_DIR, filename)
                    loader = PyMuPDFLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source_file"] = filename
                        doc.metadata["doc_type"] = "policy"
                    documents.extend(docs)
        else:
            print(f"Directory {POLICY_DIR} does not exist.")
            
        if os.path.exists(KNOWLEDGE_BASE_DIR):
            for filename in os.listdir(KNOWLEDGE_BASE_DIR):
                if filename.endswith(".txt"):
                    file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source_file"] = filename
                        doc.metadata["doc_type"] = "knowledge_base"
                    documents.extend(docs)
        else:
            print(f"Directory {KNOWLEDGE_BASE_DIR} does not exist.")
            
        print(f"Loaded {len(documents)} pages/documents in total.")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Splits documents into smaller semantic chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", r"(?<=\. )", " ", ""],
            is_separator_regex=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks.")
        return chunks

    def build_index(self, force_rebuild=False):
        """Builds or loads the FAISS and BM25 index."""
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(BM25_INDEX_PATH) and not force_rebuild:
            print("Loading existing FAISS and BM25 indexes...")
            self.vector_store = FAISS.load_local(
                FAISS_INDEX_PATH, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            with open(BM25_INDEX_PATH, 'rb') as f:
                self.bm25_retriever = pickle.load(f)
        else:
            print("Building new FAISS and BM25 indexes...")
            documents = self.load_documents()
            if not documents:
                print("No documents found to build index.")
                return
                
            chunks = self.chunk_documents(documents)

            # Build a cosine-similarity index (IndexFlatIP on normalized
            # vectors is mathematically identical to cosine similarity).
            dimension = len(self.embeddings.embed_query("test"))
            index = faiss.IndexFlatIP(dimension)
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            self.vector_store.add_documents(chunks)
            os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
            self.vector_store.save_local(FAISS_INDEX_PATH)
            
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            with open(BM25_INDEX_PATH, 'wb') as f:
                pickle.dump(self.bm25_retriever, f)
                
            print("FAISS and BM25 indexes built and saved.")

    def _reciprocal_rank_fusion(
        self, results_list: List[List[Document]], rrf_k: int = 60
    ) -> List[Document]:
        """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

        Score for each document: sum over all lists of 1 / (rrf_k + rank + 1)
        Documents appearing in multiple lists receive a cumulative score boost.

        Args:
            results_list: A list of ranked document lists (e.g. [faiss_docs, bm25_docs]).
            rrf_k: Damping constant (default 60 per the original RRF paper).

        Returns:
            Deduplicated documents sorted by descending RRF score.
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for results in results_list:
            for rank, doc in enumerate(results):
                key = doc.page_content
                doc_map[key] = doc
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[key] for key in sorted_keys]

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Hybrid retrieval using RRF fusion + Cross-Encoder re-ranking.

        Pipeline:
          1. FAISS semantic search  (top-10)
          2. BM25 keyword search    (top-10)
          3. RRF merges & re-orders the two candidate pools
          4. Cross-Encoder re-ranks the fused list
          5. Return top-k documents
        """
        if not self.vector_store or not self.bm25_retriever:
            self.build_index()

        if not self.vector_store:
            return []

        self.bm25_retriever.k = 10

        vector_docs = self.vector_store.as_retriever(search_kwargs={"k": 10}).invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # --- Stage 1: RRF fusion (replaces naive concatenation) ---
        fused_docs = self._reciprocal_rank_fusion([vector_docs, bm25_docs])

        if not fused_docs:
            return []

        # --- Stage 2: Cross-Encoder re-ranking ---
        pairs = [[query, doc.page_content] for doc in fused_docs]
        scores = self.cross_encoder.predict(pairs)

        sorted_docs = sorted(zip(scores, fused_docs), key=lambda x: x[0], reverse=True)

        return [doc for score, doc in sorted_docs[:k]]

# Singleton instance for easy importing
rag_pipeline = RAGPipeline()

if __name__ == "__main__":
    # Test the pipeline
    pipeline = RAGPipeline()
    pipeline.build_index(force_rebuild=True)
    
    print("\n--- Testing Retrieval ---")
    query = "What happens if a customer files a chargeback after 120 days?"
    results = pipeline.retrieve(query)
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Source File : {doc.metadata.get('source_file', 'Unknown')}")
        print(f"Doc Type    : {doc.metadata.get('doc_type', 'Unknown')}")
        print(f"Page        : {doc.metadata.get('page', 'N/A')}")
        print(f"Content     : {doc.page_content}")
