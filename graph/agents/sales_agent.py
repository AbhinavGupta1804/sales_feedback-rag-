import json
from rag.rag3 import get_retriever
from graph.llm import initialize_openai
llm = initialize_openai()
    
def sales_analyzer_agent(state):
    """
    Optimized Agent: One LLM call, targeted RAG context per category, 
    and clean JSON output for the dashboard.
    """
    print("--- CALLING OPTIMIZED SALES ANALYZER (JSON OUTPUT) ---")
    
    transcript = state.get("transcript", "")

    # 1. GATHER ALL CONTEXT (Free & Local Retrieval)
    # Each category pulls strictly from its assigned 'Expert Aisle'
    well_ctx = get_retriever(transcript, "Section 4: Tone, Empathy & Communication")
    improve_ctx = get_retriever(transcript, "Section 1: Discovery & Needs Analysis (SPIN, Sandler & Challenger)")
    action_ctx = get_retriever(transcript, "Section 3: Closing Techniques & Commitment")
    followup_ctx = get_retriever(transcript, "Section 5: Follow-Up Strategies & Cadence")

    # 2. BUNDLE EVERYTHING INTO A STRUCTURED PROMPT
    prompt = f"""
    You are an expert Sales Operations Analyst. Analyze the transcript using ONLY the expert rules provided for each specific category.

    ### KNOWLEDGE CATEGORIES:
    - FOR 'WHAT WENT WELL': Use these Tone & Empathy rules: {well_ctx}
    - FOR 'WHAT TO IMPROVE': Use these Discovery & Objection rules: {improve_ctx} 
    - FOR 'RECOMMENDED NEXT ACTIONS': Use these Closing & Follow-up rules: {action_ctx} {followup_ctx}

    ### TRANSCRIPT:
    {transcript}

    ### INSTRUCTIONS:
    Analyze the call and generate a JSON object with exactly these 4 keys:
    1. "rep_performance": Evaluate the sales representative’s overall performance in the following call transcript.

Use this scale:
9-10: Excellent — strong discovery, clear value communication, effective objection handling, engaging tone, and clear next steps.
7-8: Good — relevant questions and value explanation, but some missed opportunities or weaker closing.
5-6: Average — basic flow, limited depth, inconsistent engagement or direction.
3-4: Weak — poor questioning, unclear value, missed objections, little progress.
1-2: Very poor — unprepared, disengaged, no structure or progress.
Return:
Score: [number from 1 to 10]  
Reason: [one short sentence explaining the score]

2. "what_went_well": Provide exactly 4 points (1–2 lines each) identifying successes using the Tone & Empathy context.
Return as a JSON array of 4 strings.

3. "what_to_improve": Provide exactly 4 points (1–2 lines each) identifying missed tactics using the Discovery & Objection context.
Return as a JSON array of 4 strings.

4. "recommended_next_actions": Provide exactly 4 points (1–2 lines each).
- 2 using Follow-up context
- 2 using Closing context
Return as a JSON array of 4 strings.

### OUTPUT FORMAT (VALID JSON ONLY):
{{
  "rep_performance": "string",
  "what_went_well": ["string", "string", "string", "string"],
  "what_to_improve": ["string", "string", "string", "string"],
  "recommended_next_actions": ["string", "string", "string", "string"]
}}

    You must return ONLY a valid JSON object. Do not include markdown code blocks, conversational filler, or any text outside the JSON.
    """

    # 3. SINGLE LLM CALL
    response = llm.invoke(prompt)
    raw_text = response.content.strip()

    # 4. ROBUST JSON CLEANING
    if raw_text.startswith("```json"):
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
    elif raw_text.startswith("```"):
        raw_text = raw_text.replace("```", "").strip()

    try:
        data = json.loads(raw_text)
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        # Fallback values
        data = {{
            "rep_performance": "Analysis failed.",
            "what_went_well": "N/A",
            "what_to_improve": "N/A",
            "recommended_next_actions": "N/A"
        }}

    return {
        "rep_performance": data.get("rep_performance"),
        "what_went_well": data.get("what_went_well"),
        "what_to_improve": data.get("what_to_improve"),
        "recommended_next_actions": data.get("recommended_next_actions")
    }












    
    
    
    
 