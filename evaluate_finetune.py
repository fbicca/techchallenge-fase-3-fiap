#!/usr/bin/env python3
"""
Métricas para verificar se o fine-tuning surtiu efeito no domínio MedPT.

1) Perda / perplexidade (teacher forcing): NLL médio por token no texto completo
   formatado como no treino (system + user + assistant). Quanto menor, melhor o
   modelo prevê a resposta de referência do dataset.

2) Opcional (--with-rouge): ROUGE-L F1 entre resposta gerada e resposta de referência
   (subconjunto por ser mais lento).

Recomendação: use uma fatia de validação DISJUNTA do treino, por exemplo treino com
  --max_samples 5000 nos primeiros exemplos e avaliação com:
  --eval-start 5000 --eval-max-samples 500

Exemplo:
  python evaluate_finetune.py \\
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \\
    --adapter-path artifacts/finetuned/qwen-fast \\
    --eval-start 5000 \\
    --eval-max-samples 500 \\
    --output-json artifacts/eval/metrics_qwen_fast.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

# Reutiliza governança e formatação idênticas ao treino
from train_medpt import load_and_prepare_dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover

    def tqdm(x, **kwargs):  # type: ignore[misc]
        return x


def _train_ns_from_eval_args(args: argparse.Namespace) -> Namespace:
    """Namespace compatível com load_and_prepare_dataset (mesmos flags do train_medpt)."""
    return Namespace(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        max_samples=None,
        no_preprocessing=args.no_preprocessing,
        no_anonymization=args.no_anonymization,
        no_curation=args.no_curation,
        min_question_chars=args.min_question_chars,
        min_answer_chars=args.min_answer_chars,
        max_question_answer_chars=args.max_question_answer_chars,
        no_metadata_in_prompt=args.no_metadata_in_prompt,
    )


def _load_model(base_id: str, adapter_path: Path | None, *, trust_remote_code: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs: dict[str, Any] = {"torch_dtype": dtype, "device_map": "auto", "trust_remote_code": trust_remote_code}
    else:
        kwargs = {"torch_dtype": torch.float32, "device_map": None, "trust_remote_code": trust_remote_code}

    model = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
    if adapter_path is not None and adapter_path.is_dir():
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    if not torch.cuda.is_available():
        model = model.to(torch.device("cpu"))
    return tokenizer, model


def _device_for_batch(model: torch.nn.Module) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        p = next(model.parameters())
        return p.device
    except StopIteration:
        return torch.device("cpu")


def compute_nll_and_ppl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ds: Dataset,
    *,
    max_seq_length: int,
    batch_size: int,
) -> dict[str, float]:
    """NLL médio por token (natural log) e perplexidade exp(nll)."""

    def tok(batch: dict[str, list]) -> dict[str, Any]:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        return enc

    ds_tok = ds.map(tok, batched=True, remove_columns=ds.column_names)
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)
    loader = DataLoader(ds_tok, batch_size=batch_size, shuffle=False, collate_fn=collator)
    device = _device_for_batch(model)

    sum_nll = 0.0
    n_tokens = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="NLL", leave=False):
            labels = batch["labels"]
            valid = (labels != -100).sum().item()
            if valid == 0:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = float(out.loss.item())
            sum_nll += loss * valid
            n_tokens += valid

    if n_tokens == 0:
        return {"mean_nll_natural": float("nan"), "perplexity": float("nan"), "num_tokens": 0.0}

    mean_nll = sum_nll / n_tokens
    return {
        "mean_nll_natural": mean_nll,
        "perplexity": math.exp(mean_nll),
        "num_tokens": float(n_tokens),
    }


def _build_prompt_for_generation(
    tokenizer: AutoTokenizer,
    question: str,
    answer_ref: str,
    *,
    include_meta: bool,
) -> str:
    """Prompt só até antes da resposta do assistente (para gerar e comparar com answer_ref)."""
    from train_medpt import SYSTEM_PROMPT, build_user_content, fallback_format_text, messages_for_row

    row = {"question": question, "answer": answer_ref}
    user = build_user_content(row, include_meta)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return (
        f"### Instrução\n{SYSTEM_PROMPT}\n\n### Entrada\n{user}\n\n### Resposta\n"
    )


def compute_rouge_subset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ds: Dataset,
    *,
    max_new_tokens: int,
    max_samples: int,
    include_meta: bool,
) -> dict[str, float]:
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise ImportError("Instale rouge-score: pip install rouge-score") from e

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    device = _device_for_batch(model)
    n = min(max_samples, len(ds))
    scores: list[float] = []

    model.eval()
    for i in tqdm(range(n), desc="ROUGE (geração)", leave=False):
        q = str(ds[i]["question"])
        ref = str(ds[i]["answer"]).strip()
        prompt = _build_prompt_for_generation(tokenizer, q, ref, include_meta=include_meta)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        r = scorer.score(ref, gen)["rougeL"].fmeasure
        scores.append(r)

    return {
        "rougeL_f1_mean": sum(scores) / len(scores) if scores else float("nan"),
        "rouge_num_samples": float(len(scores)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Avaliar fine-tuning MedPT (NLL/PPL e opcional ROUGE-L)",
        epilog=(
            "Exemplo (uma linha; obrigatório usar os nomes --flag antes de cada valor):\n"
            "  python evaluate_finetune.py "
            "--model-name-or-path Qwen/Qwen2.5-1.5B-Instruct "
            "--adapter-path artifacts/finetuned/qwen-fast "
            "--eval-start 5000 --eval-max-samples 500 "
            "--output-json artifacts/eval/metrics_qwen_fast.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model_name_or_path",
        "--model-name-or-path",
        "--model",
        dest="model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="ID Hugging Face ou pasta local do modelo base.",
    )
    p.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Pasta com adapter PEFT. Se omitido, avalia só o modelo base.",
    )
    p.add_argument(
        "--dataset_name",
        "--dataset-name",
        dest="dataset_name",
        type=str,
        default="AKCIT/MedPT",
    )
    p.add_argument(
        "--dataset_split",
        "--dataset-split",
        dest="dataset_split",
        type=str,
        default="train",
    )
    p.add_argument(
        "--eval-start",
        type=int,
        default=0,
        help="Índice inicial no split (fatia de validação).",
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=500,
        help="Número de exemplos na fatia [eval_start, eval_start + eval_max_samples).",
    )
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--output-json", type=str, default=None, help="Grava métricas neste ficheiro JSON.")
    p.add_argument("--no-metadata-in-prompt", action="store_true")
    p.add_argument("--no-preprocessing", action="store_true")
    p.add_argument("--no-anonymization", action="store_true")
    p.add_argument("--no-curation", action="store_true")
    p.add_argument("--min-question-chars", type=int, default=10)
    p.add_argument("--min-answer-chars", type=int, default=10)
    p.add_argument("--max-question-answer-chars", type=int, default=50000)
    p.add_argument(
        "--with-rouge",
        action="store_true",
        help="Calcula ROUGE-L após geração (lento; requer rouge-score).",
    )
    p.add_argument("--rouge-max-samples", type=int, default=50)
    p.add_argument("--rouge-max-new-tokens", type=int, default=256)
    p.add_argument("--skip-base", action="store_true", help="Não avalia o modelo base (só fine-tuned).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    row_end = args.eval_start + args.eval_max_samples
    train_ns = _train_ns_from_eval_args(args)

    base = args.model_name_or_path
    adapter = Path(args.adapter_path).resolve() if args.adapter_path else None
    if args.adapter_path and (adapter is None or not adapter.is_dir()):
        print(f"Erro: adapter não encontrado: {args.adapter_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Carregando tokenizer e dataset avaliação [{args.eval_start}, {row_end})...")
    tokenizer0 = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer0.pad_token is None:
        tokenizer0.pad_token = tokenizer0.eos_token
    ds_text, stats = load_and_prepare_dataset(
        train_ns,
        tokenizer0,
        row_start=args.eval_start,
        row_end=row_end,
        keep_qa_columns=args.with_rouge,
    )
    print(json.dumps({"governance_stats": stats, "eval_rows": len(ds_text)}, ensure_ascii=False, indent=2))
    if len(ds_text) == 0:
        print("Nenhum exemplo na fatia de avaliação.", file=sys.stderr)
        sys.exit(1)

    results: dict[str, Any] = {
        "base_model": base,
        "adapter_path": str(adapter) if adapter else None,
        "eval_slice": [args.eval_start, row_end],
        "governance_stats": stats,
        "max_seq_length": args.max_seq_length,
    }

    if not args.skip_base:
        print("Avaliando modelo BASE (sem adapter)...")
        tok_b, model_b = _load_model(base, None)
        results["base"] = compute_nll_and_ppl(
            model_b,
            tok_b,
            ds_text,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
        )
        print(json.dumps({"base": results["base"]}, ensure_ascii=False, indent=2))
        del model_b
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if adapter:
        print("Avaliando modelo FINE-TUNED (base + adapter)...")
        tok_f, model_f = _load_model(base, adapter)
        ds_for_nll = ds_text.remove_columns([c for c in ds_text.column_names if c != "text"]) if "text" in ds_text.column_names else ds_text
        results["finetuned"] = compute_nll_and_ppl(
            model_f,
            tok_f,
            ds_for_nll,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
        )
        print(json.dumps({"finetuned": results["finetuned"]}, ensure_ascii=False, indent=2))

        if not args.skip_base and "base" in results:
            b = results["base"]["mean_nll_natural"]
            f = results["finetuned"]["mean_nll_natural"]
            if not (math.isnan(b) or math.isnan(f)):
                results["delta_mean_nll"] = f - b
                results["relative_ppl_ratio"] = results["finetuned"]["perplexity"] / results["base"]["perplexity"]
                if f < b:
                    results["interpretation"] = "Fine-tuning reduziu NLL médio nesta fatia: melhor encaixe às respostas MedPT formatadas como no treino."
                else:
                    results["interpretation"] = (
                        "NLL médio não diminuiu nesta fatia (poucas épocas, subconjunto pequeno ou desalinhamento "
                        "treino/aval). Considere mais dados de treino ou fatia de validação mais representativa."
                    )

        if args.with_rouge:
            include_meta = not args.no_metadata_in_prompt
            print(f"ROUGE-L em até {args.rouge_max_samples} exemplos (geração)...")
            tok_rb, model_rb = _load_model(base, None)
            try:
                results["rouge_base"] = compute_rouge_subset(
                    model_rb,
                    tok_rb,
                    ds_text,
                    max_new_tokens=args.rouge_max_new_tokens,
                    max_samples=args.rouge_max_samples,
                    include_meta=include_meta,
                )
                results["rouge_finetuned"] = compute_rouge_subset(
                    model_f,
                    tok_f,
                    ds_text,
                    max_new_tokens=args.rouge_max_new_tokens,
                    max_samples=args.rouge_max_samples,
                    include_meta=include_meta,
                )
                rb = results["rouge_base"]["rougeL_f1_mean"]
                rf = results["rouge_finetuned"]["rougeL_f1_mean"]
                if not (math.isnan(rb) or math.isnan(rf)):
                    results["delta_rougeL_f1"] = rf - rb
            finally:
                del model_rb
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del model_f
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"Métricas gravadas em: {out_path.resolve()}")


if __name__ == "__main__":
    main()
