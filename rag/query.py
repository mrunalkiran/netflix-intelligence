import os
import sys
import chromadb
import openai
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHROMA_DIR = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
COLLECTION = "netflix_facts"

# ── Retrieve ──────────────────────────────────────────────────────────────────
def retrieve(question, top_k=5):
    collection = chroma.get_collection(name=COLLECTION)
    response = client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    )
    q_embedding = response.data[0].embedding
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k
    )
    return results['documents'][0]

