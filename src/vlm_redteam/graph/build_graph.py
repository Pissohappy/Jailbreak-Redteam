"""Graph builder for LangGraph."""

from langgraph.graph import START, END, StateGraph

from .state import BeamState
from .nodes.expand import expand_node
from .nodes.execute import execute_node
from .nodes.judge import judge_node
from .nodes.select import select_node


def compile_graph():
    """Compile a minimal linear graph to validate wiring."""
    graph = StateGraph(BeamState)
    graph.add_node("expand", expand_node)
    graph.add_node("execute", execute_node)
    graph.add_node("judge", judge_node)
    graph.add_node("select", select_node)

    graph.add_edge(START, "expand")
    graph.add_edge("expand", "execute")
    graph.add_edge("execute", "judge")
    graph.add_edge("judge", "select")
    graph.add_edge("select", END)

    return graph.compile()
