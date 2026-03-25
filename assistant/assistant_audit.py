"""Logging estruturado para rastreamento e auditoria do assistente (Etapa 3)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def assistant_audit_path(artifacts_root: str) -> str:
    return os.path.join(artifacts_root, "logs", "assistant_audit.jsonl")


def append_assistant_audit(
    artifacts_root: str,
    event: str,
    payload: dict[str, Any],
    *,
    enabled: bool = True,
) -> None:
    """
    Uma linha JSON por evento (append-only), compatível com análise e auditoria.

    Eventos sugeridos: ASSISTANT_RUN_START, ASSISTANT_RUN_END, ASSISTANT_SAFETY_FLAGS.
    """
    if not enabled:
        return
    os.makedirs(os.path.join(artifacts_root, "logs"), exist_ok=True)
    line = json.dumps(
        {"ts": _utc_iso(), "event": event, **payload},
        ensure_ascii=False,
        default=str,
    )
    path = assistant_audit_path(artifacts_root)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
