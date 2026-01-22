from typing import TypedDict, Optional, Dict

class State(TypedDict):
    transcript: str
    call_summary: str
    customer_intent: str
    rep_performance: str
    what_went_well: str
    what_to_improve: str
    objection_analysis: str
    recommended_next_actions: str

