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

