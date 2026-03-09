from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(streaming=False):
    return ChatGoogleGenerativeAI( 
        model="gemini-3.1-flash-lite-preview",
        temperature=0,
        streaming=streaming,
        max_output_tokens=512
    )