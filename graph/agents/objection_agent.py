import json
from rag.rag3 import get_retriever
from graph.llm import initialize_openai
llm = initialize_openai()

def objection_expert(state):
    """
    Expert Agent that analyzes the transcript for missed objection-handling 
    opportunities using filtered RAG context.
    """
    print("--- CALLING OBJECTION EXPERT ---")
    
    # Extract the transcript from the current state
    transcript = state.get("transcript", "")
    if not transcript:
        return {"objection_analysis": "Error: No transcript found in state."}

    # 1. Targeted Retrieval from Section 2
    # This ensures we only use objection-specific coaching points
    section_to_search = "Section 2: Objection Handling (Validate, Isolate, Reframe)"
    
    try:
        # Retrieve relevant coaching chunks from your FAISS vector store
        relevant_docs = get_retriever(transcript, section_to_search)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
    except Exception as e:
        print(f"Error during retrieval: {e}")
        context = "No specific context retrieved."

    # 2. Expert Prompting
    prompt = f"""
    You are a Sales Performance Coach. Analyze the transcript using the provided Knowledge Base.
    
    KNOWLEDGE BASE:
    {context}

    TRANSCRIPT:
    {transcript}

    INSTRUCTIONS:
    - Provide a concise summary (maximum 5 lines).
    - Focus only on the 2 most critical missed opportunities.
    - Do not use bullet points or stars.
    - Use direct language: "You missed X; instead, say Y."

    OUTPUT FORMAT:
    One short paragraph or exactly two numbered points. No other text.
    """

    # 3. Generate the analysis using Gemini
    response = llm.invoke(prompt)
    
    # Update the state with the result
    return {
        "objection_analysis": response.content.strip()
    } 
    
    