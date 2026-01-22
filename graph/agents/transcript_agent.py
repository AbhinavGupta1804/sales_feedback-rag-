import json
from rag.rag3 import get_retriever
from graph.llm import initialize_openai
llm = initialize_openai()


def transcript_analyzer(state):
    transcript = state.get("transcript","")
    
    prompt = f"""You are an expert Sales Operations Analyst. Your task is to analyze
    the provided sales call transcript and extract specific metadata to help a sales 
    manager understand the conversation at a glance.

### INSTRUCTIONS:
1. Read the transcript carefully.
2. Generate a "call_summary": A concise 3-4 sentence overview of the main topics
discussed, the flow of the conversation, and the outcome.
3. Generate a "customer_intent": Identify the primary goal or motivation of the
customer (e.g., "Exploring pricing for a large-scale migration,"
"Seeking technical clarification on API limits," or "Ready to purchase but needs a discount").

### OUTPUT FORMAT:
You must return ONLY a valid JSON object. Do not include any conversational filler or markdown formatting outside of the JSON block.

{{
  "call_summary": "string",
  "customer_intent": "string"
}}

### TRANSCRIPT:
{transcript}"""

    response = llm.invoke(prompt)
    raw_text = response.content.strip()
    
    # Cleaning the response in case the LLM includes markdown backticks
    if raw_text.startswith("```json"):
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
    elif raw_text.startswith("```"):
        raw_text = raw_text.replace("```", "").strip()

    try:
        # Parse the JSON string into a dictionary
        data = json.loads(raw_text)
        
        # Return the dictionary to update the Graph State
        return {
            "call_summary": data.get("call_summary", "Summary not generated."),
            "customer_intent": data.get("customer_intent", "Intent not identified.")
        }
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        # Fallback in case of JSON failure
        return {
            "call_summary": "Error parsing summary.",
            "customer_intent": "Error parsing intent."
        }
        
        
        
        
  