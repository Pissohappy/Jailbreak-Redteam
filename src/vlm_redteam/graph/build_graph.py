"""Graph builder for LangGraph."""

from langgraph.graph import END, START, StateGraph

from .nodes.execute import execute_node
from .nodes.expand import expand_node
from .nodes.judge import judge_node
from .nodes.select import select_node
from .state import BeamState, GraphState


def _route_after_select(state: GraphState | BeamState) -> str:
    if isinstance(state, dict):
        return "end" if bool(state.get("done", False)) else "expand"
    return "end"


def compile_graph(checkpointer=None):
    """Compile graph with SELECT_BEAM conditional loop control."""

    graph = StateGraph(GraphState)
    graph.add_node("expand", expand_node)
    graph.add_node("execute", execute_node)
    graph.add_node("judge", judge_node)
    graph.add_node("select", select_node)

    graph.add_edge(START, "expand")
    graph.add_edge("expand", "execute")
    graph.add_edge("execute", "judge")
    graph.add_edge("judge", "select")
    graph.add_conditional_edges(
        "select",
        _route_after_select,
        {"expand": "expand", "end": END},
    )

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
