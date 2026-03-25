"""Estado tipado dos fluxos LangGraph (modularização explícita)."""
from __future__ import annotations

from typing import TypedDict


class ContextAssistantState(TypedDict, total=False):
    """Fluxo: prontuário → LLM → auditoria de segurança."""

    question: str
    patient_context: str
    answer: str
    safety_flags: list[str]


class SqlAssistantState(TypedDict, total=False):
    """Fluxo: agente SQL → auditoria de segurança."""

    question: str
    answer: str
    safety_flags: list[str]
