"""Graph builder for LangGraph."""

from langgraph.graph import END, START, StateGraph

from .nodes.execute import execute_node
from .nodes.expand import expand_node
from .nodes.judge import judge_node
from .nodes.select import select_node
from .state import BeamState


def compile_graph():
    """Compile a minimal linear graph with EXPAND wired in."""

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
