"""
Pré-processamento, anonimização e curadoria aplicados antes do fine-tuning.

Objetivo: camada explícita de governança de dados alinhada a boas práticas (LGPD / ética em NLP)
e reprodutibilidade acadêmica. O dataset MedPT já passou por curadoria na origem; aqui
reforçamos limpeza, máscaras de PII e filtros de qualidade mínima.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

# Colunas textuais do MedPT (e análogos) sujeitas a pré-processamento e anonimização.
TEXT_COLUMNS = (
    "question",
    "answer",
    "condition",
    "medical_specialty",
    "question_type",
)


def preprocess_text(text: str) -> str:
    """
    Normalização e limpeza leve: Unicode, espaços, quebras de linha.
    Não remove conteúdo semântico; evita ruído de encoding e whitespace.
    """
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def anonymize_text(text: str) -> str:
    """
    Máscaras heurísticas para possíveis dados identificáveis (PII) em texto livre PT-BR.

    Limitações: não substitui avaliação de risco por especialista nem NER clínico;
    reduz vazamento óbvio (e-mail, telefone, CPF, CEP, URLs).
    """
    if not text:
        return text

    # E-mails
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        text,
        flags=re.IGNORECASE,
    )
    # URLs
    text = re.sub(r"https?://[^\s\]\)]+", "[URL]", text, flags=re.IGNORECASE)
    # CPF (com ou sem pontuação)
    text = re.sub(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", "[CPF]", text)
    # Telefones BR comuns (fixo/celular)
    text = re.sub(
        r"\b(?:\+55\s*)?(?:\(?\d{2}\)?\s*)?\d{4,5}[-\s]?\d{4}\b",
        "[TELEFONE]",
        text,
    )
    # CEP
    text = re.sub(r"\b\d{5}-?\d{3}\b", "[CEP]", text)

    return text


def cleanse_row(
    row: dict[str, Any],
    *,
    apply_preprocessing: bool,
    apply_anonymization: bool,
) -> dict[str, Any]:
    """Aplica pré-processamento e/ou anonimização às colunas textuais conhecidas."""
    out = dict(row)
    for col in TEXT_COLUMNS:
        if col not in out or out[col] is None:
            continue
        v = str(out[col])
        if apply_preprocessing:
            v = preprocess_text(v)
        if apply_anonymization:
            v = anonymize_text(v)
        out[col] = v
    return out


def cleanse_batched(
    batch: dict[str, list],
    *,
    apply_preprocessing: bool,
    apply_anonymization: bool,
) -> dict[str, list]:
    """Versão batched para `datasets.map`."""
    keys = list(batch.keys())
    if not keys:
        return batch
    n = len(batch[keys[0]])
    for i in range(n):
        row = {k: batch[k][i] for k in keys}
        row = cleanse_row(
            row,
            apply_preprocessing=apply_preprocessing,
            apply_anonymization=apply_anonymization,
        )
        for k in keys:
            if k in row:
                batch[k][i] = row[k]
    return batch


def passes_curation(
    row: dict[str, Any],
    *,
    min_question_chars: int,
    min_answer_chars: int,
    max_question_answer_chars: int,
) -> bool:
    """
    Curadoria: descarta exemplos vazios, muito curtos ou excessivamente longos (custo/tokens).
    """
    q = row.get("question")
    a = row.get("answer")
    if q is None or a is None:
        return False
    qs = str(q).strip()
    ans = str(a).strip()
    if len(qs) < min_question_chars or len(ans) < min_answer_chars:
        return False
    if len(qs) + len(ans) > max_question_answer_chars:
        return False
    return True