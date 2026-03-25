"""
Limites de atuação, sinais de risco na resposta e bloco de explainability (fontes).
"""
from __future__ import annotations

import re
from typing import Any

# Texto injetado como mensagem de sistema — define fronteiras do assistente (Etapa 3).
SYSTEM_LIMITS_AND_SAFETY = (
    "Você é um assistente de apoio à informação em saúde em português do Brasil. "
    "Regras obrigatórias:\n"
    "- Nunca prescreva medicamentos, doses, nem substitua um médico ou outro profissional habilitado.\n"
    "- Não forneça diagnóstico definitivo nem plano terapêutico fechado; use linguagem condicional e "
    "recomende sempre validação com profissional de saúde presencial ou telemedicina regulamentada.\n"
    "- Se a pergunta pedir receita, posologia ou conduta exclusiva do médico, recuse a solicitação e "
    "explique que isso exige avaliação humana.\n"
    "- Baseie-se no contexto do prontuário fornecido quando existir; se faltar informação, diga que não há "
    "dado suficiente no registro.\n"
    "- Não incentive automedicação nem condutas de risco.\n"
    "Use o contexto do prontuário quando relevante. Tom informativo e educativo."
)


def format_explainability_block(
    *,
    mode: str,
    db_file: str | None,
    base_model: str,
    adapter_path: str,
    sql_tables: str | None = None,
) -> str:
    """Bloco legível para o utilizador: fontes e limites (explainability)."""
    lines = [
        "",
        "---",
        "### Fontes e transparência (explainability)",
        "",
        "- **Modelo de linguagem:** geração com base **" + base_model + "** + adapter fine-tuned em **" + adapter_path + "** (domínio MedPT na Etapa 1).",
    ]
    if mode == "context":
        lines.append(
            "- **Dados estruturados:** resumo de **todos os pacientes** presentes no prontuário demo (SQLite), "
            "sem filtro por um único identificador."
        )
        if db_file:
            lines.append(f"  - Arquivo da base: `{db_file}`")
        lines.append(
            "- **Ordem de uso:** os dados do prontuário foram lidos antes da geração e incluídos no prompt; "
            "a resposta deve ser interpretada junto com esse contexto, não como laudo isolado."
        )
    elif mode == "sql":
        lines.append(
            "- **Dados estruturados:** consultas via agente LangChain sobre SQLite (tabelas de prontuário demo)."
        )
        if sql_tables:
            lines.append(f"  - Tabelas disponíveis ao agente: {sql_tables}")
        if db_file:
            lines.append(f"  - Arquivo: `{db_file}`")
    lines.extend(
        [
            "",
            "### Limites (segurança e validação humana)",
            "",
            "- Este assistente **não prescreve** e **não substitui** julgamento clínico.",
            "- Respostas são **assistivas**; decisões de tratamento exigem **validação por profissional habilitado**.",
            "",
        ]
    )
    return "\n".join(lines)


def scan_response_for_safety_flags(text: str) -> list[str]:
    """
    Heurística simples para auditoria: padrões que podem indicar prescrição ou conduta indevida.
    Não bloqueia a resposta; apenas registra alertas para revisão humana.
    """
    flags: list[str] = []
    t = text.lower()
    # Posologia numérica explícita
    if re.search(r"\b\d+\s*(mg|mcg|g|ml|comprimidos?|cápsulas?)\b", t, re.I):
        flags.append("possivel_mencao_posologia_numerica")
    if re.search(r"\btome\s+\d+", t, re.I):
        flags.append("possivel_instrucao_tome_dose")
    if re.search(r"\breceita\b|\bposologia\b|\bpresc(?:reva|rever)\b", t, re.I):
        flags.append("palavras_chave_prescricao")
    return flags
