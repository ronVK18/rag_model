from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
load_dotenv()
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
)

NEO4J_URI = "neo4j+s://9fe536fd.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "PBoAD8GGRlDGhtoQJtRg47u7mSBZiL8LqEZpCBRBepc"

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
app = FastAPI()
class TransactionModel(BaseModel):
    transactions: list[str]
class Query(BaseModel):
    query:str

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


@app.get("/")
def read_root():
    return {"Hello": "World"}
@app.post("/add")
def read_item(inputData:TransactionModel):
    transaction=inputData.dict()
    transaction=transaction['transactions']
    # store_transactions(transaction)
    return {
        "sucess":True
    }

@app.post("/detect")
def read_query(q:Query):
    query=q.dict()
    query=query['query']
    iso_forest_model = train_anomaly_detector()
    if iso_forest_model:
        result = check_transaction_anomaly(query, iso_forest_model)
        print(result)
    return {
        "result":result
    }