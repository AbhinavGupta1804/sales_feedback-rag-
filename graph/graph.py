from langgraph.graph import StateGraph, START, END
from graph.state import State
from graph.agents.transcript_agent import transcript_analyzer
from graph.agents.sales_agent import  sales_analyzer_agent
from graph.agents.objection_agent import  objection_expert



graph = StateGraph(State)

graph.add_node("Transcript analyzer",transcript_analyzer)
graph.add_node("Sales Coach",sales_analyzer_agent)
graph.add_node("Objection expert",objection_expert)


graph.add_edge(START,"Transcript analyzer")
graph.add_edge(START,"Sales Coach")
graph.add_edge(START,"Objection expert")
graph.add_edge("Transcript analyzer",END)
graph.add_edge("Sales Coach",END)
graph.add_edge("Objection expert",END)

final_graph = graph.compile()
