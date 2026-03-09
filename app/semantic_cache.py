from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from app.config import CACHE_DB

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_cache_db():
    return Chroma(
        collection_name="semantic_cache",
        persist_directory=CACHE_DB,
        embedding_function=embeddings
    )
def search_cache(question: str):
    db = get_cache_db()

    results = db.similarity_search_with_score(question, k=1)

    if not results:
        print("No results in cache DB")
        return None

    doc, distance = results[0]

    print("Cached question:", doc.page_content)
    print("Distance:", distance)

    if distance < 0.35:
        print("Cache HIT")
        return doc.metadata["answer"]

    print("Cache MISS due to threshold")
    return None

def save_cache(question: str, answer: str):
    db = get_cache_db()   # <-- CALL FUNCTION ()

    doc = Document(
        page_content=question,
        metadata={"answer": answer}
    )

    db.add_documents([doc])