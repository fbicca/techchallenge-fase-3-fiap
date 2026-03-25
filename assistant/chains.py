"""Pipelines LangChain (legado): encadeamento simples; o fluxo principal usa LangGraph em `langgraph_graphs`."""
from __future__ import annotations

from typing import Any

from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnableLambda

from .database import fetch_all_patients_context_text
from .langgraph_graphs import compile_context_assistant_graph
from .prompts import build_context_prompt_string
from .sql_agent_factory import build_sql_agent

__all__ = ["build_patient_context_chain", "build_sql_agent"]


def build_patient_context_chain(
    tokenizer: Any,
    llm: HuggingFacePipeline,
    *,
    db_path: Any = None,
):
    """
    Encadeamento compatível com LangChain: delega ao grafo LangGraph e devolve só o texto da resposta.

    Entrada: {\"question\": str}
    Saída: str (apenas `answer`).
    """
    graph = compile_context_assistant_graph(tokenizer, llm, db_path=db_path)

    def _invoke(inputs: dict[str, Any]) -> str:
        out = graph.invoke(
            {
                "question": inputs["question"],
            }
        )
        return str(out.get("answer", ""))

    return RunnableLambda(_invoke)


def build_patient_context_chain_linear(
    tokenizer: Any,
    llm: HuggingFacePipeline,
    *,
    db_path: Any = None,
):
    """
    Cadeia linear pura LangChain (sem LangGraph), útil se `langgraph` não estiver instalado.
    """
    from langchain_core.runnables import RunnableSequence

    def add_context(inputs: dict[str, Any]) -> dict[str, Any]:
        ctx = fetch_all_patients_context_text(db_path=db_path)
        return {**inputs, "patient_context": ctx}

    def to_prompt(inputs: dict[str, Any]) -> str:
        return build_context_prompt_string(
            tokenizer,
            patient_context=inputs["patient_context"],
            question=inputs["question"],
        )

    return RunnableLambda(add_context) | RunnableLambda(to_prompt) | llm
