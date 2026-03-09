from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm import get_llm,get_text_response
from app.semantic_cache import search_cache, save_cache
from app.vector_db import get_retriever,get_db,rerank_documents
from app.planner import classify_question
from app.memory import get_memory, add_to_memory
from app.user_memory import update_memory

def format_doc(docs):
    return "\n\n".join([d.page_content for d in docs])


def ask_question(question:str):

    if not question.strip():
        return {"error": "Question cannot be empty"}
    print('---------n--------')
    print('User Question:')
    print(question)
    print ('----------------n-----------')
    normalized = question.strip().lower()

    print('before search cache')
    cached = search_cache(normalized)
    print('After search cache')
    if cached:
        print("Answer from cache\n")
        return {"answer":cached,"cached":True}
    
    category = classify_question(question)
    print("Planner:",category)

    if category == "FACTUAL":
        k = 3
    elif category == "ANALYTICAL":
        k = 6

    elif category == "MULTI_POLICY":
        k = 8

    else:
        k = 5
    
    retriever = get_retriever(k)
    docs = retriever.invoke(question)

    print('Reranker start')
    docs = rerank_documents(question, docs, top_k=5)

    print("📄 RETRIEVED DOCUMENTS:\n")
    for i, d in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print("Content Preview:")
        print(d.page_content[:500])  # only first 500 chars
        print("Metadata:", d.metadata)

    context = format_doc(docs)

    print("\n🧠 CONTEXT SENT TO LLM (first 1000 chars):\n")
    print(context[:1000])

    print('Getting from memory')
    memory_context = get_memory()

    prompt = ChatPromptTemplate.from_template("""
You are a compliance research assistant.
                                              
Conversation History:
{memory}

Use ONLY the provided context.

If the answer is directly stated, quote it.
If the answer requires explanation, logically synthesize it from the context.

Important Rules:

1. The answer may require combining information from multiple policy sections.

2. Even if the answer is not explicitly written in one place,
   explain it using the relevant policies found in the context.

3. If financial or procurement policies reference regulations
   such as 2 CFR 200, explain how related personnel or administrative
   policies support compliance.

4. Only say "Not found in documents" if NO relevant policies
   appear in the context.

When explaining, follow this structure:

1. Identify relevant policy sections
2. Explain what control they establish
3. Explain how that control reduces risk or ensures compliance

Context:
{context}

Question:
{question}
""")

    llm = get_llm()

    chain = prompt | llm | StrOutputParser()

    draft_answer = chain.invoke({
        "memory": memory_context,
        "context": context,
        "question": question
    })

    reflection_prompt = f"""
Evaluate this RAG answer.

Question:
{question}

Answer:
{draft_answer}

Classify answer quality:

GOOD = Answer supported by context

PARTIAL = Some information missing

BAD = Context insufficient

Reply ONLY:

GOOD
PARTIAL
BAD
"""

    reflection = get_text_response(llm, reflection_prompt)

    MAX_RETRY = 1

    if reflection in ["YES","PARTIAL","BAD"] or "Not found" in draft_answer:
        print("Reflection:",reflection)
        print("Reflection triggered – Retrieving more context")
        db=get_db()
        retriever = db.as_retriever(search_kwargs={"k": 8})
        new_docs = retriever.invoke(question)

        new_docs = rerank_documents(question,new_docs,top_k=5)
        all_docs = docs + new_docs

        unique_docs = {doc.page_content: doc for doc in all_docs}
        docs = list(unique_docs.values())[:8]

        context = format_doc(docs)

        final_answer = chain.invoke({
            "context": context,
            "question": question
        })
    else:
        final_answer = draft_answer
        print("No Reflection triggered")

    print("\n LLM ANSWER:\n")
    print(final_answer)
    confidence_prompt = f"""
Question:
{question}

Answer:
{final_answer}

Rate confidence from 0 to 1.

Only return number.
"""
    confidence = get_text_response(llm, confidence_prompt)
    print('Added to memory')
    add_to_memory(question, final_answer)
    print('Updated to memory')
    update_memory(question)
    
    print ('before saved')
    if "Not found" not in final_answer:
        save_cache(normalized, final_answer)
        print('cache saved')
    else:
        print("Not caching failed answer")

    return {"answer": final_answer, "confidence": confidence, "cached": False}
