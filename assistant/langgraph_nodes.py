"""Nós do grafo (uma responsabilidade por função — fácil de testar e estender)."""
from __future__ import annotations

from typing import Any

from langchain_community.llms import HuggingFacePipeline

from .database import fetch_all_patients_context_text
from .langgraph_state import ContextAssistantState, SqlAssistantState
from .prompts import build_context_prompt_string
from .security import scan_response_for_safety_flags


def make_node_fetch_prontuario(db_path: Any) -> Any:
    def fetch_prontuario(state: ContextAssistantState) -> dict[str, Any]:
        ctx = fetch_all_patients_context_text(db_path=db_path)
        return {"patient_context": ctx}

    return fetch_prontuario


def make_node_generate(llm: HuggingFacePipeline, tokenizer: Any) -> Any:
    def generate(state: ContextAssistantState) -> dict[str, Any]:
        prompt = build_context_prompt_string(
            tokenizer,
            patient_context=state["patient_context"],
            question=state["question"],
        )
        raw = llm.invoke(prompt)
        text = raw if isinstance(raw, str) else str(raw)
        return {"answer": text}

    return generate


def node_safety_scan_context(state: ContextAssistantState) -> dict[str, Any]:
    flags = scan_response_for_safety_flags(state.get("answer", ""))
    return {"safety_flags": flags}


def make_node_sql_agent(agent: Any) -> Any:
    def run_sql_agent(state: SqlAssistantState) -> dict[str, Any]:
        result = agent.invoke({"input": state["question"]})
        out = result.get("output", result)
        text = out if isinstance(out, str) else str(out)
        return {"answer": text}

    return run_sql_agent


def node_safety_scan_sql(state: SqlAssistantState) -> dict[str, Any]:
    flags = scan_response_for_safety_flags(state.get("answer", ""))
    return {"safety_flags": flags}
