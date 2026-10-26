from typing import TypedDict, List, Optional
from langchain_core.documents import Document

class InvestigationState(TypedDict):
    # Input
    query: str
    
    # State accumulated
    transaction_id: Optional[str]
    claim_id: Optional[str]
    structured_data_summary: str
    retrieved_documents: List[Document]
    
    # Execution logic
    revision_number: int
    human_approved: bool
    
    # Outputs
    analysis_draft: str
    confidence_score: float
    critic_feedback: str
    final_report: str
