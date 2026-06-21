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

# ── Answer ────────────────────────────────────────────────────────────────────
def ask(question):
    chunks = retrieve(question)
    context = "\n".join(f"- {c}" for c in chunks)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Netflix content analyst. "
                    "Answer questions using ONLY the facts below. "
                    "Be concise and insightful.\n\n"
                    f"Facts:\n{context}"
                )
            },
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content, chunks

if __name__ == '__main__':
    questions = [
        "Which country produces the most Netflix content?",
        "What is the most common genre on Netflix?",
        "How has Netflix grown over the years?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        answer, _ = ask(q)
        print(f"A: {answer}")