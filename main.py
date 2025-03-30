from openai import OpenAI
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Sample past transactions (Replace with actual user transactions)
transactions = [
    "User A bought groceries worth $50 at Walmart.",
    "User A spent $20 on coffee at Starbucks.",
    "User A paid $60 for dinner at a restaurant.",
    "User A paid $48 for groceries at Target.",
    "User A purchased electronics worth $200 from Best Buy.",
    "User A transferred $500 to a friend for rent.",
    "User A spent $35 on a movie ticket and snacks.",
    "User A paid $80 for a gym membership.",
    "User A withdrew $100 from an ATM in New York.",
    "User A booked a hotel room for $150 in Chicago."
]


# Generate embeddings
def get_embeddings(texts):
    response = client.embeddings.create(
        input=texts,
        model="BAAI/bge-multilingual-gemma2",
    )
    return np.array([data.embedding for data in response.data])

transaction_embeddings = get_embeddings(transactions)

# Train Isolation Forest for anomaly detection
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(transaction_embeddings)

# Anomaly detection function
def check_transaction_anomaly(new_transaction):
    new_embedding = get_embeddings([new_transaction])[0]
    prediction = iso_forest.predict([new_embedding])

    if prediction[0] == 1:
        return "Matched (Normal Transaction)"
    else:
        return "Not Matched (Anomalous Transaction 🚨)"

# Example usage
new_transaction = "User A purchased 3 Bitcoin worth $90,000 from a crypto exchange."
result = check_transaction_anomaly(new_transaction)
print(result)

