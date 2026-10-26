import re
from typing import Literal

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import InvestigationState
from src.rag_pipeline import rag_pipeline
from src.tools import fetch_transaction, fetch_claim, fetch_merchant, run_sql_query

# Configure the LLM
# Assuming Ollama is running locally with llama3 or qwen2.5
llm = ChatOllama(model="llama3", temperature=0)

def planner_node(state: InvestigationState) -> dict:
    """Plans the investigation steps and extracts entities like transaction ID."""
    print("--- PLANNER AGENT ---")
    
    prompt = f"""You are a planner for a financial dispute investigation. 
    Analyze the user query and extract any Transaction ID (e.g., TXN00123) or Claim ID (e.g., CLM0012).
    Also, list the steps needed to investigate.
    
    User Query: {state['query']}
    
    Output format:
    TRANSACTION_ID: [id or None]
    CLAIM_ID: [id or None]
    STEPS:
    1. ...
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    
    # Simple extraction
    tx_match = re.search(r'TRANSACTION_ID:\s*(TXN\d+)', content)
    claim_match = re.search(r'CLAIM_ID:\s*(CLM\d+)', content)
    
    tx_id = tx_match.group(1) if tx_match else None
    claim_id = claim_match.group(1) if claim_match else None
    
    return {"transaction_id": tx_id, "claim_id": claim_id}

def retrieval_node(state: InvestigationState) -> dict:
    """Retrieves relevant policy documents based on the query."""
    print("--- RETRIEVAL AGENT ---")
    docs = rag_pipeline.retrieve(state['query'], k=3)
    return {"retrieved_documents": docs}

def analysis_node(state: InvestigationState) -> dict:
    """Gathers structured data and analyzes it against policies."""
    print("--- ANALYSIS AGENT ---")
    
    # Use create_react_agent to autonomously gather structured data via tools
    from langgraph.prebuilt import create_react_agent
    tools = [fetch_transaction, fetch_claim, fetch_merchant, run_sql_query]
    
    # We use a tool-calling agent (which utilizes .bind_tools under the hood)
    data_agent = create_react_agent(llm, tools=tools)
    
    # Provide the planner's extracted entities as context to the agent
    prompt = f"Gather all necessary structured data from the database to investigate this user query: {state['query']}"
    if state.get("transaction_id"):
        prompt += f"\nHint: The planner identified transaction ID: {state['transaction_id']}"
    if state.get("claim_id"):
        prompt += f"\nHint: The planner identified claim ID: {state['claim_id']}"
        
    print("Running Data Gathering Sub-Agent...")
    try:
        response = data_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        structured_summary = response["messages"][-1].content
    except Exception as e:
        print(f"Data agent error: {e}")
        structured_summary = f"Failed to gather data automatically. Error: {e}"
        
    if not structured_summary or len(structured_summary) < 10:
        structured_summary = "No structured data found for the given entities."
        
    policy_texts = "\n\n".join([
        f"[Source: {doc.metadata.get('source_file', 'Unknown')}]\n{doc.page_content}" 
        for doc in state.get('retrieved_documents', [])
    ])
    
    prompt = f"""You are a financial dispute analyst.
    Based on the structured data and retrieved policies, write an analysis of the dispute.
    
    When writing your Risk Assessment, you must explicitly state if the transaction was flagged as a statistical anomaly (`is_anomaly`) or if the customer made rapid successive transactions (`time_since_last_txn_hours`). Also note any warnings about messy merchant names.
    
    User Query: {state['query']}
    
    Structured Data:
    {structured_summary}
    
    Policy Documents:
    {policy_texts}
    
    Generate an Analysis Draft covering Risk Assessment and Likely Outcome.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "structured_data_summary": structured_summary,
        "analysis_draft": response.content
    }

def critic_node(state: InvestigationState) -> dict:
    """Reviews the analysis and generates a final report and confidence score."""
    print("--- CRITIC AGENT ---")
    
    prompt = f"""You are a senior dispute auditor.
    Review the following analysis draft against the structured data and policies.
    Check if conclusions are supported. Ensure the Analyst explicitly addressed any anomalous transaction data (like high Z-scores) or fuzzy-matched merchant names. If they missed it, penalize the confidence score.
    
    Draft: {state['analysis_draft']}
    
    Generate a final investigation report formatted in Markdown with these sections:
    - Executive Summary
    - Case Details
    - Retrieved Evidence (Cite your sources)
    - Risk Assessment
    - Likely Outcome
    - Recommended Action
    - Confidence Score (0-100%)
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    score_match = re.search(r'Confidence Score:.*?(\d+)', response.content, re.IGNORECASE)
    score = float(score_match.group(1)) if score_match else 70.0
    
    return {
        "final_report": response.content,
        "critic_feedback": "Review complete.",
        "confidence_score": score,
        "revision_number": state.get("revision_number", 0) + 1
    }

def human_review_node(state: InvestigationState) -> dict:
    """Pauses for human review."""
    print("--- HUMAN REVIEW NODE ---")
    return {"human_approved": True}

def route_after_critic(state: InvestigationState) -> str:
    score = state.get("confidence_score", 0.0)
    rev = state.get("revision_number", 0)
    
    if score < 80.0 and rev < 3:
        print(f"Confidence {score} < 80. Routing back to Analyst (Revision {rev+1})")
        return "analyst"
    
    return "human_review"

# Build the Graph
workflow = StateGraph(InvestigationState)

workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieval_node)
workflow.add_node("analyst", analysis_node)
workflow.add_node("critic", critic_node)
workflow.add_node("human_review", human_review_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "analyst")
workflow.add_edge("analyst", "critic")
workflow.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "analyst": "analyst",
        "human_review": "human_review"
    }
)
workflow.add_edge("human_review", END)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory, interrupt_before=["human_review"])

if __name__ == "__main__":
    import sys
    print("Testing LangGraph Workflow...")
    
    # Allow passing a query via terminal argument, otherwise use a default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Investigate transaction TXN00010. Customer claims fraud."
        print(f"No query provided. Using default: '{query}'")
        print(f"Usage: python -m src.graph \"<your custom query here>\"")
        print("-" * 40)
        
    test_state = {"query": query}
    
    try:
        final_state = app_graph.invoke(test_state)
        print("\n\n=== FINAL REPORT ===\n")
        print(final_state.get('final_report', 'No report generated.'))
    except Exception as e:
        print(f"Error during graph execution: {e}")
        print("Make sure ollama is running and 'llama3' is pulled.")
