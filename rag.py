from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
# Load environment variables
load_dotenv()
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
)

# Neo4j connection details
NEO4J_URI = "neo4j+s://9fe536fd.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "PBoAD8GGRlDGhtoQJtRg47u7mSBZiL8LqEZpCBRBepc"

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Generate embeddings
def get_embeddings(texts):
    response = client.embeddings.create(
        input=texts,
        model="BAAI/bge-multilingual-gemma2",
    )
    return np.array([data.embedding for data in response.data])

# Store transaction embeddings in Neo4j
def store_transaction(tx, text, embedding):
    tx.run(
        """
        CREATE (t:Transaction {text: $text, embedding: $embedding})
        """,
        text=text, embedding=embedding.tolist()
    )

def store_transactions(transactions):
    embeddings = get_embeddings(transactions)
    with driver.session() as session:
        for text, embedding in zip(transactions, embeddings):
            session.execute_write(store_transaction, text, embedding)

transactions = [
    "User A bought groceries worth $50 at Walmart.",
    "User A spent $20 on coffee at Starbucks.",
    "User A paid $60 for dinner at a restaurant.",
    "User A purchased electronics worth $200 from Best Buy.",
    "User A transferred $500 to a friend for rent."
]

store_transactions(transactions)
print("✅ Transaction embeddings stored in Neo4j!")
# Train Isolation Forest
def get_stored_embeddings(tx):
    result = tx.run("MATCH (t:Transaction) RETURN t.text, t.embedding")
    return [(record["t.text"], np.array(record["t.embedding"])) for record in result]
def train_anomaly_detector():
    with driver.session() as session:
        stored_data = session.execute_read(get_stored_embeddings)
    
    if not stored_data:
        print("🚨 No stored transactions found!")
        return None

    embeddings = np.array([embedding for _, embedding in stored_data])
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(embeddings)
    
    return iso_forest

# Check if a new transaction is anomalous
def check_transaction_anomaly(new_transaction, iso_forest):
    new_embedding = get_embeddings([new_transaction])[0]
    prediction = iso_forest.predict([new_embedding])

    if prediction[0] == 1:
        return "Matched ✅ (Normal Transaction)"
    else:
        return "🚨 Not Matched (Anomalous Transaction)"

# Train model
iso_forest_model = train_anomaly_detector()

# Test with a new transaction
new_transaction = "User A purchased 3 Bitcoin worth $90,000 from a crypto exchange."
if iso_forest_model:
    result = check_transaction_anomaly(new_transaction, iso_forest_model)
    print(result)