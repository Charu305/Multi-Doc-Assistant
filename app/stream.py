from sse_starlette.sse import EventSourceResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from .vector_db import get_db


async def stream_answer(question: str):

    db = get_db()
    docs = db.as_retriever(search_kwargs={"k": 3}).invoke(question)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer only using the context.

Context:
{context}

Question:
{question}
"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        streaming=True
    )

    async for chunk in llm.astream(prompt):
        if chunk.content:
            yield {"data": chunk.content}

def stream_response(question: str):
    return EventSourceResponse(stream_answer(question))
