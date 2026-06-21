import os
import sys
import chromadb
import openai
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipeline.analyse import load_clean, generate_facts

# ── Config ────────────────────────────────────────────────────────────────────
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHROMA_DIR = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
COLLECTION = "netflix_facts"

# ── Embed ─────────────────────────────────────────────────────────────────────
def embed_batch(texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [r.embedding for r in response.data]

# ── Ingest ────────────────────────────────────────────────────────────────────
def ingest():
    df = load_clean()
    facts = generate_facts(df)
    print(f"Ingesting {len(facts)} facts into ChromaDB...")

    collection = chroma.get_or_create_collection(name=COLLECTION)

    # Batch embed (50 at a time)
    batch_size = 50
    all_embeddings = []
    for i in range(0, len(facts), batch_size):
        batch = facts[i:i+batch_size]
        embeddings = embed_batch(batch)
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i+batch_size, len(facts))}/{len(facts)}")

    collection.upsert(
        ids=[f"fact_{i}" for i in range(len(facts))],
        embeddings=all_embeddings,
        documents=facts
    )
    print(f"Done! {len(facts)} facts stored in ChromaDB.")

if __name__ == '__main__':
    ingest()