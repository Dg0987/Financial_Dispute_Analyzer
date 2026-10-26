import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from scipy import stats
from thefuzz import fuzz, process

# --- Config ---
NUM_TRANSACTIONS = 500
NUM_CLAIMS = 100
DATA_DIR = "data/structured"
POLICY_DIR = "data/policies"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(POLICY_DIR, exist_ok=True)

# --- 1. Generate Merchants Data ---
def generate_merchants():
    merchant_categories = ['Electronics', 'Clothing', 'Digital Goods', 'Travel', 'Groceries', 'Services']
    regions = ['NA', 'EU', 'APAC', 'LATAM']
    base_names = ['Amazon', 'Walmart', 'Target', 'Best Buy', 'Apple', 'Netflix', 'Uber', 'Airbnb']
    
    merchants = []
    for i in range(1, 51): # 50 merchants
        base = random.choice(base_names)
        # Simulate messy names for fuzzy matching later
        messy_name = f"{base} {random.choice(['Inc', 'Corp', 'LLC', 'Mktp', '*'])}" if random.random() > 0.5 else base
        merchants.append({
            'merchant_id': f'M{i:03d}',
            'raw_name': messy_name,
            'standardized_name': base, # Simulated fuzzy match result
            'category': random.choice(merchant_categories),
            'risk_score': round(random.uniform(0.1, 9.9), 1),
            'region': random.choice(regions)
        })
    df_merchants = pd.DataFrame(merchants)
    df_merchants.to_csv(os.path.join(DATA_DIR, 'merchants.csv'), index=False)
    print("merchants.csv generated with raw/standardized names.")
    return df_merchants

# --- 2. Generate Transactions Data ---
def generate_transactions(df_merchants):
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Crypto', 'Bank Transfer']
    statuses = ['Completed', 'Completed', 'Completed', 'Completed', 'Pending', 'Failed'] # Weighted
    
    transactions = []
    start_date = datetime.now() - timedelta(days=90)
    
    for i in range(1, NUM_TRANSACTIONS + 1):
        merchant = df_merchants.sample().iloc[0]
        amount = round(random.uniform(10.0, 1500.0), 2)
        # Introduce some extreme outliers for anomaly detection
        if random.random() < 0.05:
            amount = round(random.uniform(5000.0, 10000.0), 2)
            
        tx_date = start_date + timedelta(days=random.randint(0, 90), hours=random.randint(0,23), minutes=random.randint(0,59))
        
        transactions.append({
            'transaction_id': f'TXN{i:05d}',
            'customer_id': f'C{random.randint(1, 200):04d}',
            'merchant_id': merchant['merchant_id'],
            'amount': amount,
            'payment_method': random.choice(payment_methods),
            'timestamp': tx_date.strftime("%Y-%m-%d %H:%M:%S"),
            'location': merchant['region'],
            'status': random.choice(statuses)
        })
        
    df_transactions = pd.DataFrame(transactions)
    
    # --- Feature Engineering & Anomaly Detection ---
    df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'])
    df_transactions = df_transactions.sort_values(by=['customer_id', 'timestamp'])
    
    # 1. Time since last transaction
    df_transactions['time_since_last_txn_hours'] = df_transactions.groupby('customer_id')['timestamp'].diff().dt.total_seconds() / 3600
    df_transactions['time_since_last_txn_hours'] = df_transactions['time_since_last_txn_hours'].fillna(-1).round(2)
    
    # 2. Z-Score anomaly detection on amount
    df_transactions['amount_z_score'] = np.abs(stats.zscore(df_transactions['amount'])).round(2)
    df_transactions['is_anomaly'] = df_transactions['amount_z_score'] > 3
    
    df_transactions.to_csv(os.path.join(DATA_DIR, 'transactions.csv'), index=False)
    print("transactions.csv generated with feature engineering and anomalies.")
    return df_transactions

# --- 3. Generate Claims Data ---
def generate_claims(df_transactions):
    claim_types = ['Chargeback', 'Fraud', 'Refund Not Received', 'Item Not As Described', 'Unauthorized Transaction']
    claim_statuses = ['Open', 'Under Investigation', 'Resolved - Customer Won', 'Resolved - Merchant Won']
    
    claims = []
    # Pick a random sample of transactions to have claims
    claimed_txns = df_transactions.sample(n=NUM_CLAIMS)
    
    for i, (_, txn) in enumerate(claimed_txns.iterrows(), 1):
        claim_type = random.choice(claim_types)
        
        desc_map = {
            'Chargeback': "Customer initiated a chargeback claiming they didn't recognize the charge.",
            'Fraud': "Cardholder claims card was stolen and used without permission.",
            'Refund Not Received': "Customer returned item but refund was not issued after 14 days.",
            'Item Not As Described': "The delivered product was damaged or different from the website description.",
            'Unauthorized Transaction': "Unrecognized login from a new device resulted in this purchase."
        }
        
        claims.append({
            'claim_id': f'CLM{i:04d}',
            'transaction_id': txn['transaction_id'],
            'claim_type': claim_type,
            'claim_description': desc_map[claim_type],
            'claim_status': random.choice(claim_statuses)
        })
        
    df_claims = pd.DataFrame(claims)
    df_claims.to_csv(os.path.join(DATA_DIR, 'claims.csv'), index=False)
    print("claims.csv generated.")
    return df_claims

# --- 4. Generate Synthetic PDF Policies ---
def create_pdf(filename, title, content_paragraphs):
    filepath = os.path.join(POLICY_DIR, filename)
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []
    
    Story.append(Paragraph(title, styles['Title']))
    Story.append(Spacer(1, 12))
    
    for para in content_paragraphs:
        Story.append(Paragraph(para, styles['Normal']))
        Story.append(Spacer(1, 12))
        
    doc.build(Story)
    print(f"Generated {filename}")

def generate_policies():
    policies = [
        {
            "filename": "chargeback_policy.pdf",
            "title": "Standard Chargeback Policy (2024)",
            "content": [
                "1. Introduction: This document outlines the standard procedures for handling chargebacks initiated by customers.",
                "2. Timeframes: Customers must file a chargeback within 120 days of the transaction date. Merchants have 30 days to respond to a chargeback notification with compelling evidence.",
                "3. Compelling Evidence: To successfully dispute a chargeback, merchants must provide evidence such as proof of delivery, AVS (Address Verification System) match, CVV match, or positive identification of the cardholder.",
                "4. Fraudulent Chargebacks (Friendly Fraud): If a merchant can prove the customer actually made and received the purchase, the chargeback will be reversed. Digital goods require IP logs and access records as proof of delivery.",
                "5. High Risk Categories: Electronics and Digital Goods are considered high-risk categories. Merchants in these categories must maintain a chargeback ratio below 1%. If the ratio exceeds 1%, a reserve may be held on the account.",
                "6. Resolution: If the merchant fails to provide evidence within 30 days, the customer automatically wins the dispute. If evidence is provided, the issuing bank will make a final decision within 45 days."
            ]
        },
        {
            "filename": "refund_and_return_policy.pdf",
            "title": "Merchant Refund and Return Guidelines",
            "content": [
                "1. Standard Return Window: Customers have the right to return physical items within 30 days of receipt for a full refund, provided the item is in its original condition.",
                "2. Digital Goods: Digital goods, software, and downloadable content are generally non-refundable once accessed or downloaded, unless proven to be defective.",
                "3. Return Shipping: Unless the item is defective or 'not as described', the customer is responsible for return shipping costs.",
                "4. Refund Processing Time: Once a returned item is received by the merchant, the merchant must process the refund within 5-7 business days. It may take an additional 3-5 days for the funds to appear on the customer's statement.",
                "5. Item Not As Described: If a customer claims an item is 'not as described', they must provide photographic evidence. If verified, the merchant must provide a prepaid return label and a full refund.",
                "6. Restocking Fees: Merchants may charge up to a 15% restocking fee for items that are returned opened but not defective. This must be clearly stated on the product page."
            ]
        },
        {
            "filename": "fraud_prevention_guidelines.pdf",
            "title": "Fraud Prevention and Liability Guidelines",
            "content": [
                "1. Liability Shift: For EMV (chip) card transactions, liability for counterfeit fraud shifts to the party that does not support EMV. For online transactions (Card Not Present), liability generally rests with the merchant unless 3D Secure was used.",
                "2. 3D Secure: Transactions authenticated with 3D Secure (e.g., Verified by Visa, Mastercard Identity Check) shift the liability for fraud-related chargebacks from the merchant to the issuer.",
                "3. Red Flags: Merchants should monitor for red flags such as: multiple failed payment attempts, mismatch between billing and shipping address, expedited shipping on high-value items, and unusually large orders.",
                "4. Account Takeover (ATO): If a customer claims 'Unauthorized Transaction' due to account takeover, the platform will investigate IP logs, device fingerprints, and recent password changes.",
                "5. Risk Scoring: Merchants with a risk score above 7.0 are subject to additional scrutiny. Transactions over $1000 from high-risk merchants may require manual review before settlement.",
                "6. Cryptocurrency Payments: Cryptocurrency transactions are final and irreversible. Claims regarding crypto payments must be resolved directly between the buyer and seller, except in cases of proven platform error."
            ]
        }
    ]
    
    for pol in policies:
        create_pdf(pol["filename"], pol["title"], pol["content"])

if __name__ == "__main__":
    print("Generating synthetic data...")
    df_merch = generate_merchants()
    df_txn = generate_transactions(df_merch)
    df_claim = generate_claims(df_txn)
    generate_policies()
    print("Data generation complete.")
