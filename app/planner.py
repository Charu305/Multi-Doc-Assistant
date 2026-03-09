from app.llm import get_llm,get_text_response

def classify_question(question):

    prompt = f"""
Classify this question into ONE category:

FACTUAL
ANALYTICAL
COMPARISON
MULTI_POLICY

Question:
{question}

Answer only category.
"""

    llm = get_llm()

    return get_text_response(llm, prompt)