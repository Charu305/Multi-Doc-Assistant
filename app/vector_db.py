from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from app.config import VECTOR_DB
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
import threading
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_db():
    return Chroma(
        collection_name="multi_doc_collection",
        persist_directory=VECTOR_DB,
        embedding_function=embeddings
        )

def build_hybrid_retriever(k=5):
        db=get_db()
        results = db.get()

        if not results["documents"]:
            return db.as_retriever(search_kwargs={"k": k})
             

        all_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(results["documents"], results["metadatas"])
        ]

        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = k * 2

        vector_retriever = db.as_retriever(search_kwargs={"k": k * 2})

        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25, vector_retriever],
            weights=[0.5, 0.5]
        )

        return hybrid_retriever

def get_retriever(k=5):
    return build_hybrid_retriever(k)

def rerank_documents(question, docs, top_k=5):

    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ## print ('scores',scores)
    scored_docs = list(zip(docs, scores))
    ##print('scored_docs',scored_docs)
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_k]]