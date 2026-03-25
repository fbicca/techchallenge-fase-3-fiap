"""
Compilação dos fluxos LangGraph.

- Modo *context*: fetch_prontuario → generate → safety_scan
- Modo *sql*: sql_agent → safety_scan
"""
from __future__ import annotations

from typing import Any

from langchain_community.llms import HuggingFacePipeline

from .langgraph_nodes import (
    make_node_fetch_prontuario,
    make_node_generate,
    make_node_sql_agent,
    node_safety_scan_context,
    node_safety_scan_sql,
)
from .sql_agent_factory import build_sql_agent


def _import_langgraph():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as e:
        raise ImportError(
            "Instale LangGraph: pip install langgraph (ver requirements-langchain.txt)."
        ) from e
    return END, START, StateGraph


def compile_context_assistant_graph(
    tokenizer: Any,
    llm: HuggingFacePipeline,
    *,
    db_path: Any = None,
) -> Any:
    """Grafo principal do assistente com prontuário estruturado + LLM."""
    END, START, StateGraph = _import_langgraph()

    from .langgraph_state import ContextAssistantState

    g = StateGraph(ContextAssistantState)
    g.add_node("fetch_prontuario", make_node_fetch_prontuario(db_path))
    g.add_node("generate", make_node_generate(llm, tokenizer))
    g.add_node("safety_scan", node_safety_scan_context)

    g.add_edge(START, "fetch_prontuario")
    g.add_edge("fetch_prontuario", "generate")
    g.add_edge("generate", "safety_scan")
    g.add_edge("safety_scan", END)

    return g.compile()


def compile_sql_assistant_graph(llm: HuggingFacePipeline, *, db_path: Any = None) -> Any:
    """Grafo enxuto: agente SQL + varredura de segurança."""
    END, START, StateGraph = _import_langgraph()

    from .langgraph_state import SqlAssistantState

    agent = build_sql_agent(llm, db_path=db_path)

    g = StateGraph(SqlAssistantState)
    g.add_node("sql_agent", make_node_sql_agent(agent))
    g.add_node("safety_scan", node_safety_scan_sql)

    g.add_edge(START, "sql_agent")
    g.add_edge("sql_agent", "safety_scan")
    g.add_edge("safety_scan", END)

    return g.compile()
