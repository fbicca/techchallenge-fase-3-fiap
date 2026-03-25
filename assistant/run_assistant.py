#!/usr/bin/env python3
"""
CLI do assistente médico (Etapa 2): LangChain + LangGraph + modelo fine-tuned + SQLite.

Fluxos principais implementados com **LangGraph** (`langgraph_graphs.py`): nós modulares em
`langgraph_nodes.py` e estado tipado em `langgraph_state.py`.

Etapa 3: limites de segurança, logging em assistant_audit.jsonl e explainability.

Exemplos (na pasta techchallenge-fase3, com venv ativo):

  python -m assistant.run_assistant \\
    --adapter-path artifacts/finetuned/smoke-qwen \\
    --question "Quais alergias aparecem nos prontuários e o que consta nos registros recentes?"

  python -m assistant.run_assistant --json \\
    --adapter-path artifacts/finetuned/smoke-qwen \\
    --question "Resuma o prontuário."

  python -m assistant.run_assistant --mode sql \\
    --adapter-path artifacts/finetuned/smoke-qwen \\
    --question "Liste nomes de pacientes com diabetes nas observações"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Garante raiz do projeto no path quando executado como script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.assistant_audit import append_assistant_audit
from assistant.database import ensure_database
from assistant.langgraph_graphs import compile_context_assistant_graph, compile_sql_assistant_graph
from assistant.model_loader import (
    build_langchain_llm,
    build_text_generation_pipeline,
    load_tokenizer_and_model,
)
from assistant.security import format_explainability_block


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assistente médico LangChain (modelo fine-tuned + prontuário demo)")
    p.add_argument(
        "--base-model",
        type=str,
        default=os.environ.get("HF_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        help="Mesmo modelo base usado no fine-tuning (Hub ou pasta local).",
    )
    p.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Pasta com adapter PEFT + tokenizer (ex.: artifacts/finetuned/run_...).",
    )
    p.add_argument(
        "--mode",
        choices=("context", "sql"),
        default="context",
        help="context: grafo prontuário→LLM→segurança. sql: grafo agente SQL→segurança.",
    )
    p.add_argument("--question", type=str, required=True, help="Pergunta em linguagem natural.")
    p.add_argument("--db-path", type=str, default=None, help="Caminho opcional para o arquivo SQLite.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--artifacts-root",
        type=str,
        default="artifacts",
        help="Raiz para logs de auditoria do assistente (Etapa 3).",
    )
    p.add_argument(
        "--no-assistant-audit",
        action="store_true",
        help="Não grava em artifacts/logs/assistant_audit.jsonl.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Saída em JSON (resposta + fontes + flags de segurança heurísticas).",
    )
    p.add_argument(
        "--no-explainability-footer",
        action="store_true",
        help="Não imprime o bloco markdown de fontes/limites após a resposta.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_path).resolve()
    if not adapter_path.is_dir():
        print(f"Erro: pasta do adapter não encontrada: {adapter_path}", file=sys.stderr)
        sys.exit(1)

    audit_on = not args.no_assistant_audit
    db_path_arg = Path(args.db_path).resolve() if args.db_path else None
    db_file_resolved = str(ensure_database(db_path_arg))

    append_assistant_audit(
        args.artifacts_root,
        "ASSISTANT_RUN_START",
        {
            "mode": args.mode,
            "question": args.question,
            "base_model": args.base_model,
            "adapter_path": str(adapter_path),
            "db_path": db_file_resolved,
            "pipeline": "langgraph",
        },
        enabled=audit_on,
    )

    if not args.json:
        print("Carregando modelo base + adapter PEFT...")
    tokenizer, model = load_tokenizer_and_model(args.base_model, str(adapter_path))
    pipe = build_text_generation_pipeline(tokenizer, model, max_new_tokens=args.max_new_tokens)
    llm = build_langchain_llm(pipe, model_id=args.base_model)

    db_path = db_path_arg

    if args.mode == "context":
        graph = compile_context_assistant_graph(tokenizer, llm, db_path=db_path)
        if not args.json:
            print("\n--- Modo: LangGraph (fetch_prontuario → generate → safety_scan) ---\n")
        state = graph.invoke({"question": args.question})
        out = str(state.get("answer", ""))
        flags = list(state.get("safety_flags") or [])
        footer = format_explainability_block(
            mode="context",
            db_file=db_file_resolved,
            base_model=args.base_model,
            adapter_path=str(adapter_path),
        )
        append_assistant_audit(
            args.artifacts_root,
            "ASSISTANT_SAFETY_FLAGS",
            {"flags": flags, "mode": "context"},
            enabled=audit_on,
        )
        append_assistant_audit(
            args.artifacts_root,
            "ASSISTANT_RUN_END",
            {
                "mode": "context",
                "response_chars": len(out),
                "safety_flags": flags,
            },
            enabled=audit_on,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "answer": out,
                        "sources": [
                            {
                                "type": "sqlite_prontuario",
                                "detail": "Resumo estruturado de todos os pacientes (prontuário demo)",
                                "file": db_file_resolved,
                            },
                            {
                                "type": "llm",
                                "detail": f"{args.base_model} + PEFT adapter",
                                "adapter_path": str(adapter_path),
                            },
                        ],
                        "safety_heuristic_flags": flags,
                        "limits": "prescricao_proibida_validacao_humana_obrigatoria",
                        "langgraph_nodes": ["fetch_prontuario", "generate", "safety_scan"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(out)
            if not args.no_explainability_footer:
                print(footer)
        return

    if not args.json:
        print("\n--- Modo: LangGraph (sql_agent → safety_scan) ---\n")
    graph = compile_sql_assistant_graph(llm, db_path=db_path)
    state = graph.invoke({"question": args.question})
    out = str(state.get("answer", ""))
    flags = list(state.get("safety_flags") or [])
    footer = format_explainability_block(
        mode="sql",
        db_file=db_file_resolved,
        base_model=args.base_model,
        adapter_path=str(adapter_path),
        sql_tables="pacientes, registros_clinicos, medicacoes_ativas",
    )
    append_assistant_audit(
        args.artifacts_root,
        "ASSISTANT_SAFETY_FLAGS",
        {"flags": flags, "mode": "sql"},
        enabled=audit_on,
    )
    append_assistant_audit(
        args.artifacts_root,
        "ASSISTANT_RUN_END",
        {"mode": "sql", "response_chars": len(out), "safety_flags": flags},
        enabled=audit_on,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "answer": out,
                    "sources": [
                        {
                            "type": "sqlite_via_sql_agent",
                            "detail": "LangChain create_sql_agent sobre prontuário demo",
                            "file": db_file_resolved,
                            "tables": "pacientes, registros_clinicos, medicacoes_ativas",
                        },
                        {"type": "llm", "detail": f"{args.base_model} + PEFT", "adapter_path": str(adapter_path)},
                    ],
                    "safety_heuristic_flags": flags,
                    "limits": "prescricao_proibida_validacao_humana_obrigatoria",
                    "langgraph_nodes": ["sql_agent", "safety_scan"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(out)
        if not args.no_explainability_footer:
            print(footer)


if __name__ == "__main__":
    main()
