import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATA_PATH = "data"
VECTOR_DB = "chroma_db"
CACHE_DB = "semantic_cache_db"