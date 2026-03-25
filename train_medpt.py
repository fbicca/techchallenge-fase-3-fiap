#!/usr/bin/env python3
"""
Fine-tuning supervisionado (SFT) a partir de um LLM *já pré-treinado*.

Fluxo pedagógico:
  1) Modelo base pré-treinado — pesos públicos de um LLM instrucional (ex.: Qwen2.5 1.5B Instruct),
     carregados via `AutoModelForCausalLM.from_pretrained`.
  2) Fine-tuning — ajuste supervisionado no domínio médico com o dataset MedPT (AKCIT/MedPT),
     usando PEFT (LoRA) ou QLoRA; não há treinamento do modelo “do zero”.

Este script não inicializa pesos aleatórios: sempre parte do checkpoint pré-treinado indicado em
`--model_name_or_path`.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import socket
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from getpass import getuser
from typing import Any

# Evita crash em operações não suportadas no backend MPS (Apple Silicon),
# fazendo fallback de kernels para CPU quando possível.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from data_governance import cleanse_batched, passes_curation

SYSTEM_PROMPT = (
    "Você é um assistente para esclarecimentos gerais em saúde em português do Brasil. "
    "Suas respostas são educativas e informativas e não substituem avaliação, diagnóstico ou "
    "tratamento por um profissional de saúde habilitado."
)

# Raiz local de evidências: separa manifest do pré-treinado, saída do fine-tuning e logs de auditoria.
DEFAULT_ARTIFACTS_ROOT = "artifacts"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit_log_path(artifacts_root: str) -> str:
    # Extensão .jsonl (uma linha JSON por evento) para não conflitar com *.log no .gitignore.
    return os.path.join(artifacts_root, "logs", "audit.jsonl")


def append_audit_event(
    artifacts_root: str,
    event: str,
    payload: dict[str, Any],
    *,
    enabled: bool,
) -> None:
    """Log de auditoria em JSON Lines (uma linha JSON por evento), append-only."""
    if not enabled:
        return
    os.makedirs(os.path.join(artifacts_root, "logs"), exist_ok=True)
    line = json.dumps(
        {"ts": _utc_now_iso(), "event": event, **payload},
        ensure_ascii=False,
        default=str,
    )
    path = _audit_log_path(artifacts_root)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_pretrained_manifest(
    artifacts_root: str,
    base_model_id: str,
    *,
    dataset_name: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """
    Evidência local: o que é o modelo pré-treinado (referência), sem copiar pesos multi-GB para o Git.
    Sobrescreve artifacts/pretrained/pretrained_manifest.json a cada execução (último estado).
    """
    pretrained_dir = os.path.join(artifacts_root, "pretrained")
    os.makedirs(pretrained_dir, exist_ok=True)
    manifest: dict[str, Any] = {
        "updated_at_utc": _utc_now_iso(),
        "role": "pretrained_base_reference",
        "description_pt": (
            "Identifica o checkpoint pré-treinado usado como base. Os pesos completos do LLM "
            "não são armazenados neste repositório; são obtidos via Hugging Face (ou caminho local) "
            "em tempo de execução."
        ),
        "base_model_pretrained": base_model_id,
        "dataset_finetune": dataset_name,
        "weights_location_policy": "hub_or_local_path_only_not_committed",
    }
    if extra:
        manifest["extra"] = extra
    path = os.path.join(pretrained_dir, "pretrained_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def write_evidence_json(
    output_dir: str,
    *,
    args_ns: argparse.Namespace,
    train_samples: int,
    duration_sec: float | None,
    log_history_tail: list[dict[str, Any]],
    status: str,
    error: str | None,
    governance_stats: dict[str, Any] | None = None,
) -> str:
    """Evidência local do run de fine-tuning (parâmetros, amostras, duração, últimos logs)."""
    payload: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "status": status,
        "error": error,
        "separation_note_pt": (
            "base_pretrained: referência em artifacts/pretrained/pretrained_manifest.json e Hub; "
            "finetuned_output: esta pasta contém adapters PEFT + tokenizer após o fine-tuning."
        ),
        "base_model_pretrained": args_ns.model_name_or_path,
        "finetuned_artifact_dir": os.path.abspath(output_dir),
        "dataset": {
            "name": args_ns.dataset_name,
            "split": args_ns.dataset_split,
            "train_samples_used": train_samples,
            "max_samples_cap": args_ns.max_samples,
        },
        "data_governance": governance_stats or {},
        "hyperparameters": {k: getattr(args_ns, k) for k in sorted(vars(args_ns))},
        "train_duration_seconds": duration_sec,
        "trainer_log_history_tail": log_history_tail[-30:],
    }
    path = os.path.join(output_dir, "evidence.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fine-tuning (SFT + LoRA) no MedPT a partir de um modelo instrucional já pré-treinado no Hub."
        )
    )
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help=(
            "Checkpoint do modelo BASE PRÉ-TREINADO (Hugging Face ou pasta local com pesos salvos). "
            "O fine-tuning continua a partir destes pesos; não treina do zero."
        ),
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default="AKCIT/MedPT",
        help="Nome do dataset no Hugging Face.",
    )
    p.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Split a usar (MedPT expõe principalmente 'train').",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Usa apenas os N primeiros exemplos do split (ex.: 5000 em vez da base inteira). "
            "Padrão: None = carregar todo o split."
        ),
    )
    p.add_argument(
        "--artifacts_root",
        type=str,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Pasta raiz para pretrained (manifest), finetuned (saída) e logs (auditoria).",
    )
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Nome da subpasta em <artifacts_root>/finetuned/ (default: run_<timestamp>).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Onde salvar adapter/tokenizer/evidence.json. Default: <artifacts_root>/finetuned/run_<timestamp>.",
    )
    p.add_argument(
        "--no_audit_log",
        action="store_true",
        help="Não escreve em artifacts/logs/audit.jsonl (não recomendado para entrega acadêmica).",
    )
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--use_qlora",
        action="store_true",
        help="Ativa carregamento 4-bit (bitsandbytes) + LoRA (QLoRA). Recomendado em GPUs com pouca VRAM.",
    )
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--no_metadata_in_prompt",
        action="store_true",
        help="Não inclui condition, medical_specialty e question_type no prompt (só a pergunta).",
    )
    p.add_argument(
        "--no_preprocessing",
        action="store_true",
        help="Desliga normalização Unicode / limpeza de whitespace (não recomendado).",
    )
    p.add_argument(
        "--no_anonymization",
        action="store_true",
        help="Desliga máscaras heurísticas de PII (e-mail, URL, CPF, telefone, CEP).",
    )
    p.add_argument(
        "--no_curation",
        action="store_true",
        help="Desliga filtros de tamanho mínimo/máximo em pergunta/resposta.",
    )
    p.add_argument(
        "--min_question_chars",
        type=int,
        default=10,
        help="Curadoria: tamanho mínimo da pergunta (após pré-processamento).",
    )
    p.add_argument(
        "--min_answer_chars",
        type=int,
        default=10,
        help="Curadoria: tamanho mínimo da resposta (alinhado ao dataset MedPT: respostas muito curtas são removidas na origem).",
    )
    p.add_argument(
        "--max_question_answer_chars",
        type=int,
        default=50000,
        help="Curadoria: soma máxima de caracteres pergunta+resposta (evita pares extremos).",
    )
    return p.parse_args()


def build_user_content(row: dict[str, Any], include_meta: bool) -> str:
    if not include_meta:
        return str(row["question"]).strip()
    parts: list[str] = []
    c = row.get("condition")
    s = row.get("medical_specialty")
    t = row.get("question_type")
    if c:
        parts.append(f"Condição/tema: {c}")
    if s:
        parts.append(f"Especialidade do contexto: {s}")
    if t:
        parts.append(f"Tipo de dúvida: {t}")
    meta = "\n".join(parts)
    q = str(row["question"]).strip()
    if meta:
        return f"{meta}\n\nPergunta do paciente:\n{q}"
    return q


def messages_for_row(row: dict[str, Any], include_meta: bool) -> list[dict[str, str]]:
    user = build_user_content(row, include_meta)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": str(row["answer"]).strip()},
    ]


def fallback_format_text(row: dict[str, Any], include_meta: bool) -> str:
    """Formato instrução simples quando o tokenizer não expõe chat_template."""
    user = build_user_content(row, include_meta)
    ans = str(row["answer"]).strip()
    return (
        f"### Instrução\n{SYSTEM_PROMPT}\n\n"
        f"### Entrada\n{user}\n\n"
        f"### Resposta\n{ans}"
    )


def make_format_batch_fn(tokenizer: AutoTokenizer, include_meta: bool):
    def _batch(batch: dict[str, list]) -> dict[str, list]:
        texts: list[str] = []
        n = len(batch["question"])
        for i in range(n):
            row = {k: batch[k][i] for k in batch}
            msgs = messages_for_row(row, include_meta)
            if getattr(tokenizer, "chat_template", None):
                t = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                t = fallback_format_text(row, include_meta)
            texts.append(t)
        return {"text": texts}

    return _batch


def load_and_prepare_dataset(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    *,
    row_start: int | None = None,
    row_end: int | None = None,
    keep_qa_columns: bool = False,
) -> tuple[Dataset, dict[str, Any]]:
    """
    Carrega o split, aplica governança de dados (pré-processamento, anonimização, curadoria)
    e formata para SFT.

    Parâmetros opcionais (avaliação / scripts externos):
    - row_start, row_end: fatia [row_start, row_end) sobre o split bruto (antes de max_samples).
    - keep_qa_columns: se True, mantém colunas question/answer além de text (para métricas com geração).
    """
    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    stats: dict[str, Any] = {
        "preprocessing_enabled": not args.no_preprocessing,
        "anonymization_enabled": not args.no_anonymization,
        "curation_enabled": not args.no_curation,
        "min_question_chars": args.min_question_chars,
        "min_answer_chars": args.min_answer_chars,
        "max_question_answer_chars": args.max_question_answer_chars,
    }

    if row_start is not None or row_end is not None:
        start = 0 if row_start is None else max(0, int(row_start))
        stop = len(ds) if row_end is None else int(row_end)
        stop = min(stop, len(ds))
        if start >= stop:
            raise ValueError(f"row_start ({start}) deve ser menor que row_end ({stop}).")
        ds = ds.select(range(start, stop))
        stats["row_slice"] = [start, stop]
    elif args.max_samples is not None:
        n = min(int(args.max_samples), len(ds))
        ds = ds.select(range(n))
    stats["rows_after_slice"] = len(ds)

    apply_prep = not args.no_preprocessing
    apply_anon = not args.no_anonymization
    if apply_prep or apply_anon:
        ds = ds.map(
            lambda b: cleanse_batched(
                b,
                apply_preprocessing=apply_prep,
                apply_anonymization=apply_anon,
            ),
            batched=True,
            batch_size=1000,
        )

    rows_before_filter = len(ds)
    if not args.no_curation:
        ds = ds.filter(
            lambda ex: passes_curation(
                ex,
                min_question_chars=args.min_question_chars,
                min_answer_chars=args.min_answer_chars,
                max_question_answer_chars=args.max_question_answer_chars,
            ),
        )
    stats["rows_after_governance"] = len(ds)
    stats["rows_dropped_by_curation"] = rows_before_filter - len(ds)

    include_meta = not args.no_metadata_in_prompt

    if keep_qa_columns:

        def _fmt_keep_qa(batch: dict[str, list]) -> dict[str, list]:
            texts: list[str] = []
            n = len(batch["question"])
            for i in range(n):
                row = {k: batch[k][i] for k in batch}
                msgs = messages_for_row(row, include_meta)
                if getattr(tokenizer, "chat_template", None):
                    t = tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                else:
                    t = fallback_format_text(row, include_meta)
                texts.append(t)
            return {
                "text": texts,
                "question": list(batch["question"]),
                "answer": list(batch["answer"]),
            }

        remove_cols = [c for c in ds.column_names if c not in ("question", "answer")]
        ds = ds.map(_fmt_keep_qa, batched=True, batch_size=1000, remove_columns=remove_cols)
    else:
        fn = make_format_batch_fn(tokenizer, include_meta)
        ds = ds.map(
            fn,
            batched=True,
            batch_size=1000,
            remove_columns=ds.column_names,
        )
    return ds, stats


def choose_lora_target_modules(model: AutoModelForCausalLM) -> list[str]:
    """
    Escolhe módulos LoRA conforme arquitetura do modelo base.

    - Llama/Mistral/Gemma/Qwen-like: projeções separadas (q_proj/k_proj/v_proj/o_proj + MLP).
    - Falcon: usa `query_key_value` e `dense*`.
    """
    model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "").lower()

    # Mapeamentos comuns por família.
    if model_type in {"falcon", "refinedweb", "refinedwebmodel"}:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    if model_type in {"llama", "mistral", "gemma", "qwen2", "qwen2_moe"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Fallback por inspeção de nomes para arquiteturas não mapeadas.
    linear_suffixes: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_suffixes.add(name.split(".")[-1])

    falcon_candidates = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    if any(s in linear_suffixes for s in falcon_candidates):
        return [s for s in falcon_candidates if s in linear_suffixes]

    llama_candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if any(s in linear_suffixes for s in llama_candidates):
        return [s for s in llama_candidates if s in linear_suffixes]

    raise ValueError(
        "Não foi possível inferir target_modules para LoRA nesta arquitetura. "
        f"model_type={model_type!r}, linear_suffixes_sample={sorted(list(linear_suffixes))[:20]}"
    )


def build_sft_trainer_compatible(
    *,
    model: AutoModelForCausalLM,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> SFTTrainer:
    """
    Cria SFTTrainer de forma compatível com variações de API entre versões do TRL.

    Algumas versões aceitam `dataset_text_field`/`max_seq_length`/`packing`;
    outras exigem `formatting_func` e removem esses argumentos.
    """
    sig = inspect.signature(SFTTrainer.__init__)
    accepted = set(sig.parameters.keys())

    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
    }

    if "processing_class" in accepted:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in accepted:
        kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in accepted:
        kwargs["dataset_text_field"] = "text"
    elif "formatting_func" in accepted:
        kwargs["formatting_func"] = lambda ex: ex["text"]

    if "max_seq_length" in accepted:
        kwargs["max_seq_length"] = max_seq_length
    if "packing" in accepted:
        kwargs["packing"] = False

    return SFTTrainer(**kwargs)


def _is_mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def main() -> None:
    args = parse_args()
    audit_enabled = not args.no_audit_log

    if args.output_dir is None:
        sub = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        args.output_dir = os.path.join(args.artifacts_root, "finetuned", sub)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.artifacts_root, "pretrained"), exist_ok=True)
    os.makedirs(os.path.join(args.artifacts_root, "logs"), exist_ok=True)

    torch.manual_seed(args.seed)
    mps_available = _is_mps_available()
    force_cpu_on_mps = mps_available and "falcon" in args.model_name_or_path.lower()

    # Falcon legado usa checkpoint interno sem `use_reentrant`, gerando warning no torch recente.
    # É apenas aviso de compatibilidade futura (não é a causa do crash).
    warnings.filterwarnings(
        "ignore",
        message=r"torch\.utils\.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly\.",
        category=UserWarning,
    )

    append_audit_event(
        args.artifacts_root,
        "TRAIN_START",
        {
            "hostname": socket.gethostname(),
            "user": getuser(),
            "cwd": os.getcwd(),
            "python": sys.version.split()[0],
            "torch_version": torch.__version__,
            "argv": sys.argv,
            "resolved_output_dir": os.path.abspath(args.output_dir),
            "base_model_pretrained": args.model_name_or_path,
            "dataset_finetune": args.dataset_name,
        },
        enabled=audit_enabled,
    )

    print(
        "\n=== Fine-tuning sobre modelo pré-treinado ===\n"
        f"  Modelo base (pré-treinado): {args.model_name_or_path}\n"
        f"  Dataset de ajuste fino:     {args.dataset_name} (split={args.dataset_split})\n"
        f"  Saída (fine-tuned):         {os.path.abspath(args.output_dir)}\n"
        f"  Manifest pré-treinado:      {os.path.join(args.artifacts_root, 'pretrained', 'pretrained_manifest.json')}\n"
        f"  Auditoria:                  {_audit_log_path(args.artifacts_root)}\n"
        "  O carregamento usa from_pretrained (pesos já treinados no checkpoint base escolhido).\n"
        "  Treináveis: adapters LoRA em cima da base congelada (ver print_trainable_parameters).\n"
        "============================================\n"
    )

    train_samples = 0
    log_history_tail: list[dict[str, Any]] = []
    duration_sec: float | None = None
    err_msg: str | None = None
    status = "failed"
    t0: float | None = None
    tokenizer: AutoTokenizer | None = None
    governance_stats: dict[str, Any] = {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        manifest_path = write_pretrained_manifest(
            args.artifacts_root,
            args.model_name_or_path,
            dataset_name=args.dataset_name,
            extra={
                "tokenizer_loaded_from": getattr(tokenizer, "name_or_path", None),
                "vocab_size": getattr(tokenizer, "vocab_size", None),
            },
        )
        append_audit_event(
            args.artifacts_root,
            "PRETRAINED_MANIFEST_WRITTEN",
            {"path": os.path.abspath(manifest_path)},
            enabled=audit_enabled,
        )

        train_ds, governance_stats = load_and_prepare_dataset(args, tokenizer)
        train_samples = len(train_ds)
        print(
            "Governança de dados (pré-processamento / anonimização / curadoria):\n"
            f"  {json.dumps(governance_stats, ensure_ascii=False, indent=2)}\n"
        )
        append_audit_event(
            args.artifacts_root,
            "DATA_GOVERNANCE_APPLIED",
            governance_stats,
            enabled=audit_enabled,
        )

        use_qlora = bool(args.use_qlora and torch.cuda.is_available())
        if args.use_qlora and not torch.cuda.is_available():
            print(
                "Aviso: --use_qlora foi solicitado, mas este PyTorch não possui CUDA. "
                "Seguindo com LoRA padrão (sem quantização 4-bit)."
            )

        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)
        else:
            # Sem CUDA:
            # - MPS: preferir float16 para reduzir memória.
            # - CPU: manter float32 para estabilidade numérica.
            if torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif mps_available and not force_cpu_on_mps:
                dtype = torch.float16
            else:
                dtype = torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )

        # Perfil conservador para Apple Silicon/MPS: modelos maiores podem estourar memória.
        if mps_available:
            if args.per_device_train_batch_size > 1:
                print(
                    "Ajuste automático (MPS): per_device_train_batch_size "
                    f"{args.per_device_train_batch_size} -> 1 para reduzir risco de OOM."
                )
                args.per_device_train_batch_size = 1
            if args.max_seq_length > 512:
                print(
                    "Ajuste automático (MPS): max_seq_length "
                    f"{args.max_seq_length} -> 512 para reduzir uso de memória."
                )
                args.max_seq_length = 512

        target_modules = choose_lora_target_modules(model)
        print(f"LoRA target_modules selecionados: {target_modules}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        model.enable_input_require_grads()
        model.config.use_cache = False

        if force_cpu_on_mps:
            print(
                "Aviso: MPS detectado com modelo Falcon. Para evitar falhas conhecidas no backend MPS, "
                "o treino sera executado em CPU (mais lento, porem mais estavel)."
            )

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=2,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
            lr_scheduler_type="cosine",
            report_to="none",
            seed=args.seed,
            no_cuda=force_cpu_on_mps,
            use_mps_device=not force_cpu_on_mps,
        )

        trainer = build_sft_trainer_compatible(
            model=model,
            training_args=training_args,
            train_dataset=train_ds,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )

        t0 = time.perf_counter()
        trainer.train()
        duration_sec = time.perf_counter() - t0
        log_history_tail = list(trainer.state.log_history)

        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        status = "ok"

        append_audit_event(
            args.artifacts_root,
            "TRAIN_END",
            {
                "status": "ok",
                "output_dir": os.path.abspath(args.output_dir),
                "train_samples": train_samples,
                "duration_seconds": duration_sec,
                "data_governance": governance_stats,
            },
            enabled=audit_enabled,
        )
        print(f"Treino concluído. Adapter, tokenizer e evidence.json em: {args.output_dir}")
    except Exception:
        err_msg = traceback.format_exc()
        if t0 is not None:
            duration_sec = time.perf_counter() - t0
        append_audit_event(
            args.artifacts_root,
            "TRAIN_ERROR",
            {
                "output_dir": os.path.abspath(args.output_dir),
                "train_samples": train_samples,
                "duration_seconds": duration_sec,
                "data_governance": governance_stats,
                "error": err_msg,
            },
            enabled=audit_enabled,
        )
        raise
    finally:
        write_evidence_json(
            args.output_dir,
            args_ns=args,
            train_samples=train_samples,
            duration_sec=duration_sec,
            log_history_tail=log_history_tail,
            status=status,
            error=err_msg if status != "ok" else None,
            governance_stats=governance_stats,
        )
        append_audit_event(
            args.artifacts_root,
            "EVIDENCE_WRITTEN",
            {"path": os.path.abspath(os.path.join(args.output_dir, "evidence.json")), "status": status},
            enabled=audit_enabled,
        )

    if status != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
