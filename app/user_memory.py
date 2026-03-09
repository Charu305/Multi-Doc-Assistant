import json
import os

MEMORY_FILE = "user_memory.json"


def load_memory():

    if os.path.exists(MEMORY_FILE):

        with open(MEMORY_FILE,"r") as f:
            return json.load(f)

    return {}


def save_memory(memory):

    with open(MEMORY_FILE,"w") as f:
        json.dump(memory,f)


def update_memory(question):

    memory = load_memory()

    if "topics" not in memory:
        memory["topics"] = {}

    words = question.lower().split()

    for w in words:

        if len(w) > 4:

            memory["topics"][w] = memory["topics"].get(w,0)+1

    save_memory(memory)