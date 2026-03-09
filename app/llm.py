from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(streaming=False):
    return ChatGoogleGenerativeAI( 
        model="gemini-3.1-flash-lite-preview",
        temperature=0,
        streaming=streaming,
        max_output_tokens=512
    )
def get_text_response(llm, prompt):
    """Handles both string and list content from Gemini models"""
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = " ".join([
            item["text"] if isinstance(item, dict) and "text" in item
            else str(item)
            for item in content
        ])
    return content.strip()