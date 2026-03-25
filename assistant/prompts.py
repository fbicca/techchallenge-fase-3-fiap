"""Montagem de prompts compartilhada entre LangChain (legado) e LangGraph."""
from __future__ import annotations

from typing import Any

from .security import SYSTEM_LIMITS_AND_SAFETY


def build_context_prompt_string(tokenizer: Any, *, patient_context: str, question: str) -> str:
    """Prompt completo (chat template ou fallback) para o modo contexto + prontuário."""
    messages = [
        {"role": "system", "content": SYSTEM_LIMITS_AND_SAFETY},
        {
            "role": "user",
            "content": (
                "Resumo dos dados estruturados dos prontuários (fonte: base local SQLite, todos os pacientes):\n\n"
                f"{patient_context}\n\n"
                f"Pergunta do utilizador ou do profissional:\n{question}"
            ),
        },
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        f"### Instrução\n{SYSTEM_LIMITS_AND_SAFETY}\n\n"
        f"### Contexto dos prontuários\n{patient_context}\n\n"
        f"### Pergunta\n{question}\n\n### Resposta\n"
    )
