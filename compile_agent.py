from langgraph.graph import StateGraph, END
from typing import TypedDict,Any
import pandas as pd

from tools import *


class AnalysisState(TypedDict):
    query:str
    df: pd.DataFrame
    columns: list
    should_plot : str
    code: Any
    result: Any
    reasoning: Any
    error: str
    is_new_dataset: bool
    insights: str



def data_insights_node(state: AnalysisState):
    """Generate dataset insights for new datasets"""
    if state.get("is_new_dataset", False):
        insights = data_insight_agent.invoke({"df": state["df"]})
        return {"insights": insights}
    return {"insights": state.get("insights", "")}



def classification_node(state:AnalysisState):
    """Classify if query needs visualization"""

    should_plot = query_classification_agent.invoke({"query":state["query"]})

    return {"should_plot": should_plot}


def code_generation_node(state: AnalysisState):

    """Generate code for data analysis"""

    code = pandas_writing_agent.invoke({"cols": state["columns"], "query": state["query"]})


    return {"code": code}


def execution_node(state :AnalysisState):
        
        """Execute the generated code"""
    
        result = execution_tool.invoke({"code": state["code"], "df": state["df"], "should_plot": state["should_plot"]})

        return {"result" : result}



def reasoning_node(state: AnalysisState):

    """Generate reasoning about the result"""

    reasoning = reasoning_agent.invoke({"query": state["query"],"result": state["result"]})

    return {"reasoning" : reasoning}


def should_continue_after_insights(state: AnalysisState):
    """Decide if we should continue to query processing or just return insights"""
    if state.get("is_new_dataset", False) and not state.get("query"):
        return "end_with_insights"
    return "classify"

def should_continue_after_execution(state: AnalysisState):
    """Decide if we should continue or end after execution"""
    if isinstance(state.get("result"), str) and "Error" in state.get("result", ""):
        return "error"
    return "reasoning"




workflow = StateGraph(AnalysisState)


workflow.add_node("insights", data_insights_node)
workflow.add_node("classify", classification_node)
workflow.add_node("generate_code", code_generation_node)
workflow.add_node("execute",execution_node)
workflow.add_node("reason", reasoning_node)




workflow.set_entry_point("insights")
workflow.add_conditional_edges(
    "insights",
    should_continue_after_insights,
    {
        "classify": "classify",
        "end_with_insights": END
    }
)
workflow.add_edge("classify", "generate_code")
workflow.add_edge("generate_code", "execute")
workflow.add_conditional_edges(
    "execute",
    should_continue_after_execution,
    {
        "reasoning": "reason",
        "error": END
    }
)

workflow.add_edge("reason",END)




app = workflow.compile()




# # Add this method to get Mermaid config
# def get_mermaid_config():
#     return app.get_graph().draw_mermaid()

# # Usage
# mermaid_config = get_mermaid_config()
# print(mermaid_config)





