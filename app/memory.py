conversation_memory = []

def add_to_memory(question, answer):

    conversation_memory.append({
        "question": question,
        "answer": answer
    })

    # Keep last 5 only
    if len(conversation_memory) > 5:
        conversation_memory.pop(0)


def get_memory():

    text = ""

    for item in conversation_memory:

        text += f"""
Previous Question:
{item['question']}

Previous Answer:
{item['answer']}

"""

    return text