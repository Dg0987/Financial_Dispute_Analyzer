import os
import pandas as pd
from tabulate import tabulate
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Import project components
from src.rag_pipeline import rag_pipeline
from src.graph import app_graph

# Initialize LLM judge
llm_judge = ChatOllama(model="llama3", temperature=0)

# Ground truth dataset
EVALUATION_DATASET = [
    {
        "query": "What happens if a customer files a chargeback after 120 days?",
        "expected_source": "chargeback_policy.pdf"
    },
    {
        "query": "How many days does a merchant have to process a refund?",
        "expected_source": "refund_and_return_policy.pdf"
    },
    {
        "query": "Who is liable for a fraudulent EMV chip card transaction?",
        "expected_source": "fraud_prevention_guidelines.pdf"
    },
    {
        "query": "Are digital goods generally refundable?",
        "expected_source": "refund_and_return_policy.pdf"
    },
    {
        "query": "What is the acceptable chargeback ratio for high risk categories?",
        "expected_source": "chargeback_policy.pdf"
    }
]

def calculate_retrieval_metrics(query, expected_source, k=3):
    """Evaluates RAG retrieval based on expected source document."""
    docs = rag_pipeline.retrieve(query, k=k)
    
    hit = 0
    mrr = 0.0
    
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', '')
        if expected_source in source:
            hit = 1
            mrr = 1.0 / (i + 1)
            break
            
    return hit, mrr, docs

def llm_as_judge(query, retrieved_docs, final_report):
    """Uses Ollama to score Faithfulness and Relevance (1-10)."""
    context_text = "\n".join([d.page_content for d in retrieved_docs])
    
    prompt = f"""You are an impartial judge evaluating an AI agent.
    Score the agent's 'Final Report' based on the 'User Query' and 'Retrieved Context'.
    
    User Query: {query}
    
    Retrieved Context:
    {context_text}
    
    Final Report:
    {final_report}
    
    Provide two scores from 1 to 10:
    1. Faithfulness: Is the report strictly based on the Retrieved Context without hallucinating facts? (1-10)
    2. Relevance: Does the report directly answer the User Query? (1-10)
    
    Output ONLY in this exact format:
    FAITHFULNESS: [Score]
    RELEVANCE: [Score]
    """
    
    try:
        response = llm_judge.invoke([HumanMessage(content=prompt)])
        content = response.content
        
        faith_match = re.search(r'FAITHFULNESS:\s*(\d+)', content)
        rel_match = re.search(r'RELEVANCE:\s*(\d+)', content)
        
        faithfulness = int(faith_match.group(1)) if faith_match else 0
        relevance = int(rel_match.group(1)) if rel_match else 0
        
        # Normalize to 0.0 - 1.0
        return faithfulness / 10.0, relevance / 10.0
    except Exception as e:
        print(f"Error during LLM judging: {e}")
        return 0.0, 0.0

def run_evaluation():
    print("Initializing FAISS Vector Store...")
    rag_pipeline.build_index()
    
    results = []
    
    print(f"\nRunning evaluation on {len(EVALUATION_DATASET)} questions...\n")
    
    for i, item in enumerate(EVALUATION_DATASET, 1):
        query = item['query']
        expected_source = item['expected_source']
        print(f"[{i}/{len(EVALUATION_DATASET)}] Query: {query}")
        
        # 1. Evaluate Retrieval
        hit, mrr, retrieved_docs = calculate_retrieval_metrics(query, expected_source)
        
        # 2. Evaluate Agent Generation
        print("  -> Generating agent response...")
        try:
            # We pass a simple state. The planner will run but fail to find a TXN_ID, which is fine for general policy questions.
            import uuid
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            final_state = app_graph.invoke({"query": query}, config=config)
            final_report = final_state.get('final_report', '')
        except Exception as e:
            print(f"  -> Agent Error: {e}")
            final_report = "Error generating report."
            
        # 3. LLM as Judge
        print("  -> Judging response...")
        faithfulness, relevance = llm_as_judge(query, retrieved_docs, final_report)
        
        results.append({
            "Query": query[:40] + "...",
            "Hit Rate": hit,
            "MRR": round(mrr, 2),
            "Faithfulness": faithfulness,
            "Relevance": relevance
        })
        
    # Calculate Averages
    df = pd.DataFrame(results)
    
    avg_hit = df["Hit Rate"].mean()
    avg_mrr = df["MRR"].mean()
    avg_faith = df["Faithfulness"].mean()
    avg_rel = df["Relevance"].mean()
    overall_avg = (avg_hit + avg_mrr + avg_faith + avg_rel) / 4
    
    # Add an average row
    avg_row = pd.DataFrame([{
        "Query": "**AVERAGE SCORE**",
        "Hit Rate": round(avg_hit, 2),
        "MRR": round(avg_mrr, 2),
        "Faithfulness": round(avg_faith, 2),
        "Relevance": round(avg_rel, 2)
    }])
    df = pd.concat([df, avg_row], ignore_index=True)
    
    print("\n=== EVALUATION RESULTS ===")
    print(tabulate(df, headers='keys', tablefmt='github', showindex=False))
    print(f"\nOverall System Score: {overall_avg:.2f} / 1.00\n")
    
    # Save to CSV
    os.makedirs("data", exist_ok=True)
    csv_path = "data/evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    run_evaluation()
