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
def generate_title_facts(df):
    title_facts = []
    for _, row in df.iterrows():
        if pd.isna(row['title']):
            continue
        fact = (
            f"{row['title']} is a {row['type']} "
            f"{'released' if row['type'] == 'Movie' else 'that aired'} in {int(row['release_year']) if pd.notna(row['release_year']) else 'unknown year'}. "
            f"It is categorized under {row['listed_in'] if pd.notna(row['listed_in']) else 'unknown genres'}. "
            f"It is rated {row['rating'] if pd.notna(row['rating']) else 'unrated'} "
            f"and is a {row['country'].split(',')[0].strip() if pd.notna(row['country']) else 'unknown'} production. "
            f"Description: {row['description'] if pd.notna(row['description']) else 'No description available.'}"
        )
        title_facts.append(fact)
    return title_facts

def ingest():
    import pandas as pd
    df = load_clean()
    facts = generate_facts(df)
    title_facts = generate_title_facts(df)
    all_facts = facts + title_facts
    print(f"Ingesting {len(all_facts)} facts into ChromaDB...")

    collection = chroma.get_or_create_collection(name=COLLECTION)

    # Batch embed (50 at a time)
    batch_size = 50
    all_embeddings = []
    for i in range(0, len(all_facts), batch_size):
        batch = all_facts[i:i+batch_size]
        embeddings = embed_batch(batch)
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i+batch_size, len(facts))}/{len(facts)}")

    collection.upsert(
        ids=[f"fact_{i}" for i in range(len(all_facts))],
        embeddings=all_embeddings,
        documents=all_facts
    )
    print(f"Done! {len(facts)} facts stored in ChromaDB.")

if __name__ == '__main__':
    ingest()