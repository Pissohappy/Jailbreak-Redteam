"""Graph builder for LangGraph."""

from langgraph.graph import END, START, StateGraph

from .nodes.execute import execute_target_node
from .nodes.expand import expand_node
from .nodes.judge import judge_candidates
from .nodes.select import select_beam_node
from .state import GraphState


def _route_after_select(state: GraphState) -> str:
    return "end" if state.get("done", False) else "expand"


def compile_graph():
    """Compile a loop graph: expand -> execute -> judge -> select -> (expand|end)."""

    graph = StateGraph(GraphState)
    graph.add_node("expand", expand_node)
    graph.add_node("execute", execute_target_node)
    graph.add_node("judge", judge_candidates)
    graph.add_node("select", select_beam_node)

    graph.add_edge(START, "expand")
    graph.add_edge("expand", "execute")
    graph.add_edge("execute", "judge")
    graph.add_edge("judge", "select")
    graph.add_conditional_edges(
        "select",
        _route_after_select,
        {
            "expand": "expand",
            "end": END,
        },
    )

    return graph.compile()
