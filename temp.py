import os
from openai import OpenAI
from neo4j import GraphDatabase
import numpy as np
import uuid

# Neo4j connection details
NEO4J_URI = "neo4j+s://9fe536fd.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "PBoAD8GGRlDGhtoQJtRg47u7mSBZiL8LqEZpCBRBepc"
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key="eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExNjQzMTU4NDI4MDI0NjgwODM4NiIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwMDk5MDk5MCwidXVpZCI6Ijg1YmExY2M0LTQ2MjItNGRlMS04NWNmLWVmOGQ2MDA4YTQyNiIsIm5hbWUiOiJmcmF1ZCIsImV4cGlyZXNfYXQiOiIyMDMwLTAzLTI5VDA1OjAzOjEwKzAwMDAifQ.BAEViAw8dwjM504CF--KsUbiJ_CTKUEkmFw2cGbRPco"
)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
transactions = [
    "User A bought groceries worth $50 at Walmart.",
    "User A spent $20 on coffee at Starbucks.",
    "User A paid $60 for dinner at a restaurant.",
    "User A purchased electronics worth $200 from Best Buy.",
    "User A transferred $500 to a friend for rent.",
]
def get_embeddings(texts):
    response = client.embeddings.create(
        input=texts,
        model="BAAI/bge-multilingual-gemma2",
    )
    return [data.embedding for data in response.data]

# Generate embeddings
# transaction_embeddings = get_embeddings(transactions)

# Store embeddings in Neo4j
def store_transaction(tx, text, embedding):
    tx.run(
        """
        CREATE (t:Transaction {id: $id, text: $text, embedding: $embedding})
        """,
        id=str(uuid.uuid4()), text=text, embedding=embedding
    )

# with driver.session() as session:
#     for i, text in enumerate(transactions):
#         session.write_transaction(store_transaction, text, transaction_embeddings[i])
from sklearn.metrics.pairwise import cosine_similarity

# Retrieve stored embeddings from Neo4j
def get_stored_embeddings(tx):
    result = tx.run("MATCH (t:Transaction) RETURN t.text, t.embedding")
    return [(record["t.text"], np.array(record["t.embedding"])) for record in result]

def retrieve_similar_transactions(new_transaction, threshold=0.7):
    new_embedding = get_embeddings([new_transaction])[0]

    with driver.session() as session:
        stored_data = session.read_transaction(get_stored_embeddings)

    # Compare with stored embeddings
    similarities = [
        (text, cosine_similarity([new_embedding], [embedding])[0][0])
        for text, embedding in stored_data
    ]

    # Find best match
    best_match = max(similarities, key=lambda x: x[1])

    if best_match[1] >= threshold:
        return f"Matched (Similar to: {best_match[0]})"
    else:
        return "Not Matched (Potentially Suspicious 🚨)"

# Test with a new transaction
new_transaction = "User A spent $55 on groceries at Target."
result = retrieve_similar_transactions(new_transaction)
print(result)

# print("✅ Embeddings stored in Neo4j!")