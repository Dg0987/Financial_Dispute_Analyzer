import streamlit as st
import os
import pandas as pd
from PIL import Image

# Setup page config
st.set_page_config(page_title="Financial Dispute Intelligence", layout="wide")

import uuid

# Ensure session state variables exist
if "investigation_state" not in st.session_state:
    st.session_state.investigation_state = None
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "needs_review" not in st.session_state:
    st.session_state.needs_review = False

def sidebar_nav():
    st.sidebar.title("Navigation")
    pages = [
        "1. Policy Documents",
        "2. Investigation",
        "3. Investigation Report",
        "4. Retrieved Evidence",
        "5. Reasoning Steps"
    ]
    return st.sidebar.radio("Go to:", pages)

def page_policies():
    st.title("Policy Documents")
    st.write("These are the synthetic policy documents currently loaded into the RAG system.")
    
    POLICY_DIR = "data/policies"
    if os.path.exists(POLICY_DIR):
        st.subheader("Default Policies (PDF)")
        files = os.listdir(POLICY_DIR)
        for f in files:
            if f.endswith(".pdf"):
                st.write(f"- 📄 {f}")
    else:
        st.warning("No policies found. Did you run the data generation script?")
        
    KNOWLEDGE_DIR = "knowledge_base_cleaned"
    if os.path.exists(KNOWLEDGE_DIR):
        st.subheader("Custom Knowledge Base (TXT)")
        files = os.listdir(KNOWLEDGE_DIR)
        for f in files:
            if f.endswith(".txt"):
                st.write(f"- 📝 {f}")
        
    st.write("---")
    st.subheader("Generate Data & Rebuild Index")
    if st.button("Run Data Generation"):
        with st.spinner("Generating data and policies..."):
            os.system("python src/data_generation.py")
        st.success("Data generated successfully!")
        
    if st.button("Rebuild RAG Index"):
        with st.spinner("Rebuilding FAISS index..."):
            from src.rag_pipeline import rag_pipeline
            rag_pipeline.build_index(force_rebuild=True)
        st.success("Index rebuilt successfully!")

def page_investigation():
    st.title("New Investigation")
    st.write("Ask a question about a transaction or claim to start the multi-agent investigation.")
    
    # Show a random transaction ID as a hint if data exists
    hint = "E.g., Investigate transaction TXN00010. Customer claims it was unauthorized."
    try:
        df = pd.read_csv("data/structured/transactions.csv")
        sample_txn = df['transaction_id'].sample().iloc[0]
        hint = f"E.g., Investigate transaction {sample_txn}."
    except Exception:
        pass

    query = st.chat_input(placeholder=hint)
    
    if query:
        if not query.strip():
            st.error("Please enter a query.")
            return
            
        st.session_state.agent_logs = []
        st.session_state.investigation_state = None
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.needs_review = False
        
        with st.spinner("Initializing Agents..."):
            from src.graph import app_graph
            
            initial_state = {"query": query, "revision_number": 0, "human_approved": False}
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            with st.status("Initializing Investigation Workflow...", expanded=True) as status:
                try:
                    final_state = initial_state.copy()
                    for output in app_graph.stream(initial_state, config=config):
                        for node_name, state_update in output.items():
                            status.update(label=f"Agent '{node_name}' finished processing...", state="running")
                            st.write(f"✅ {node_name.capitalize()} Agent completed.")
                            final_state.update(state_update)
                            
                    st.session_state.investigation_state = final_state
                    
                    graph_state = app_graph.get_state(config)
                    if "human_review" in graph_state.next:
                        status.update(label="Paused for Human Review!", state="error", expanded=True)
                        st.session_state.needs_review = True
                        st.warning("⚠️ The Auditor (Critic) requested a human review before finalizing. Please review the draft.")
                    else:
                        status.update(label="Investigation Complete!", state="complete", expanded=False)
                        st.success("Analysis generated! View it in the 'Investigation Report' page.")
                        if 'final_report' in final_state:
                            st.markdown("### Preview of the Final Report")
                            st.markdown(final_state['final_report'])
                except Exception as e:
                    status.update(label="Error in execution", state="error")
                    st.error(f"Error: {e}. Ensure Ollama is running and requirements are installed.")
                    
    if st.session_state.needs_review:
        st.info("Review the draft below and approve to finalize.")
        st.markdown(f"**Current Draft:**\n{st.session_state.investigation_state.get('analysis_draft', 'N/A')}")
        if st.button("Approve & Finalize Report"):
            from src.graph import app_graph
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            with st.spinner("Resuming Graph..."):
                final_state = st.session_state.investigation_state
                for output in app_graph.stream(None, config=config):
                    for node_name, state_update in output.items():
                        final_state.update(state_update)
                st.session_state.investigation_state = final_state
                st.session_state.needs_review = False
            st.success("Report Finalized! View it in the Investigation Report tab.")

def page_report():
    st.title("Final Investigation Report")
    state = st.session_state.investigation_state
    
    if state and 'final_report' in state:
        st.markdown(state['final_report'])
        st.info(f"**Confidence Score:** {state.get('confidence_score', 'N/A')}%")
    else:
        st.warning("No report available. Please run an investigation first.")

def page_evidence():
    st.title("Retrieved Evidence")
    state = st.session_state.investigation_state
    
    if state:
        st.subheader("Structured Data Findings")
        st.text(state.get('structured_data_summary', 'No structured data retrieved.'))
        
        st.write("---")
        
        st.subheader("Retrieved Policy Documents (RAG)")
        docs = state.get('retrieved_documents', [])
        if not docs:
            st.write("No documents retrieved.")
        else:
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Source {i} - {doc.metadata.get('source', 'Unknown')}"):
                    st.write(doc.page_content)
    else:
        st.warning("No evidence available. Please run an investigation first.")

def page_reasoning():
    st.title("Agent Reasoning Steps")
    state = st.session_state.investigation_state
    
    if state:
        st.subheader("1. Planner Agent")
        st.write(f"**Extracted Transaction ID:** {state.get('transaction_id')}")
        st.write(f"**Extracted Claim ID:** {state.get('claim_id')}")
        
        st.subheader("2. Retrieval Agent")
        st.write(f"Retrieved {len(state.get('retrieved_documents', []))} document chunks from FAISS.")
        
        st.subheader("3. Analysis Agent Draft")
        st.markdown(state.get('analysis_draft', 'No draft generated.'))
        
        st.subheader("4. Critic Agent Feedback")
        st.write(state.get('critic_feedback', 'No feedback.'))
    else:
        st.warning("No reasoning available. Please run an investigation first.")

def main():
    selection = sidebar_nav()
    
    if selection == "1. Policy Documents":
        page_policies()
    elif selection == "2. Investigation":
        page_investigation()
    elif selection == "3. Investigation Report":
        page_report()
    elif selection == "4. Retrieved Evidence":
        page_evidence()
    elif selection == "5. Reasoning Steps":
        page_reasoning()

if __name__ == "__main__":
    main()
