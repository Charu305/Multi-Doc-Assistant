from app.llm import get_llm

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

    return llm.invoke(prompt).content.strip()