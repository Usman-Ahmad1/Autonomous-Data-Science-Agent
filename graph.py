import pandas as pd
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv
import os
from langgraph.graph import END
from agents.eda_agent import eda_agent

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    dataset: pd.DataFrame | None
    cleaned_dataset: pd.DataFrame | None
    next_agent: str
    cleaning_report: dict
    feature_report: dict
    eda_report: str
    eda_plot: str
    eda_complete: bool

def supervisor_node(state: AgentState):
    """
    Supervisor - Directly go to EDA_Agent
    """
    print("🔄 Supervisor: Going to EDA_Agent")
    return {
        "messages": state["messages"], 
        "next_agent": "EDA_Agent"
    }

# Build Graph - DIRECT EDA (Feature engineering is done in UI before this)
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("EDA_Agent", eda_agent)

# Set entry point
workflow.set_entry_point("supervisor")

# Define edges
workflow.add_edge("supervisor", "EDA_Agent")
workflow.add_edge("EDA_Agent", END)

# Compile the graph
graph = workflow.compile()
print("✅ Graph compiled successfully!")