import pandas as pd
import os

DATA_DIR = "data/structured"

def load_data(filename: str) -> pd.DataFrame:
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame()

df_transactions = load_data("transactions.csv")
df_claims = load_data("claims.csv")
df_merchants = load_data("merchants.csv")

def get_transaction_details(transaction_id: str) -> str:
    """Gets details for a specific transaction ID."""
    if df_transactions.empty:
        return "Transaction data not loaded."
        
    result = df_transactions[df_transactions['transaction_id'] == transaction_id]
    if result.empty:
        return f"No transaction found with ID {transaction_id}."
    
    txn = result.iloc[0]
    details = txn.to_string()
    
    # Highlight new anomaly features if present
    if 'is_anomaly' in txn and txn['is_anomaly']:
        details += f"\n\n🚨 WARNING: This transaction is flagged as a statistical anomaly!"
        details += f"\n- Z-Score: {txn.get('amount_z_score', 'N/A')}"
        
    if 'time_since_last_txn_hours' in txn and txn['time_since_last_txn_hours'] != -1:
        details += f"\n- Time since last transaction: {txn['time_since_last_txn_hours']} hours"
        
    return details

def get_claim_details(claim_id: str = None, transaction_id: str = None) -> str:
    """Gets claim details by claim_id or transaction_id."""
    if df_claims.empty:
        return "Claim data not loaded."
        
    if claim_id:
        result = df_claims[df_claims['claim_id'] == claim_id]
    elif transaction_id:
        result = df_claims[df_claims['transaction_id'] == transaction_id]
    else:
        return "Must provide either claim_id or transaction_id."
        
    if result.empty:
        return f"No claim found."
    
    return result.iloc[0].to_string()

def get_merchant_history(merchant_id: str) -> str:
    """Gets merchant risk profile and recent transaction count."""
    if df_merchants.empty or df_transactions.empty:
        return "Merchant or transaction data not loaded."
        
    merchant = df_merchants[df_merchants['merchant_id'] == merchant_id]
    if merchant.empty:
        return f"No merchant found with ID {merchant_id}."
        
    m = merchant.iloc[0]
    tx_count = len(df_transactions[df_transactions['merchant_id'] == merchant_id])
    
    details = f"Merchant ID: {merchant_id}\n"
    if 'raw_name' in m and 'standardized_name' in m:
        raw = str(m['raw_name'])
        
        # Get list of all known standardized bases to simulate entity resolution
        known_bases = df_merchants['standardized_name'].dropna().unique().tolist()
        
        # Perform real fuzzy matching
        from thefuzz import process, fuzz
        best_match, score = process.extractOne(raw, known_bases, scorer=fuzz.token_sort_ratio)
        
        details += f"Raw Name: {raw}\n"
        details += f"Resolved Name: {best_match} (Match Confidence: {score}%)\n"
        
        if score < 100:
            details += "🚨 WARNING: The merchant name was matched fuzzily. The raw string deviates from the known entity, which could indicate spoofing.\n"
    
    details += f"Category: {m['category']}\nRisk Score: {m['risk_score']}\nTotal Transactions: {tx_count}"
    
    return details

# LangChain tool wrappers will be defined in the graph module, or we can define them here.
from langchain_core.tools import tool

@tool
def fetch_transaction(transaction_id: str) -> str:
    """Use this tool ONLY when you have a specific, single Transaction ID (e.g., TXN00010) and need the details for that one specific record. Do not use for aggregations."""
    return get_transaction_details(transaction_id)
    
@tool
def fetch_claim(transaction_id: str) -> str:
    """Use this tool ONLY to fetch claim details associated with a single, known transaction ID."""
    return get_claim_details(transaction_id=transaction_id)
    
@tool
def fetch_merchant(merchant_id: str) -> str:
    """Use this tool ONLY to fetch the risk profile and history for a specific, single merchant ID (e.g., M024)."""
    return get_merchant_history(merchant_id)

import sqlite3
@tool
def run_sql_query(query: str) -> str:
    """Use this tool to execute SQLite queries for aggregations, filters, or multi-row questions (e.g., counting total disputes, finding high-risk merchants, or calculating average transaction amounts). Do NOT use this for single-record lookups if you already have the ID. Tables available: transactions, claims, merchants."""
    conn = sqlite3.connect(':memory:')
    if not df_transactions.empty: df_transactions.to_sql('transactions', conn, index=False)
    if not df_claims.empty: df_claims.to_sql('claims', conn, index=False)
    if not df_merchants.empty: df_merchants.to_sql('merchants', conn, index=False)
    
    try:
        result = pd.read_sql_query(query, conn)
        return result.to_string()
    except Exception as e:
        return f"Error executing query: {e}"
