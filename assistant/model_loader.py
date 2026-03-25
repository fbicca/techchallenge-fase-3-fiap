"""
Carrega para inferência o modelo **após fine-tuning**: base pré-treinada (Hub) + adapter PEFT (Etapa 1).

Sem o adapter em `adapter_path`, você só teria o modelo base, não o modelo ajustado no MedPT.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import Pipeline

try:  # Preferido (evita warning de depreciação do langchain_community)
    from langchain_huggingface import HuggingFacePipeline
except ImportError:  # pragma: no cover
    try:
        from langchain_community.llms import HuggingFacePipeline
    except ImportError:  # pragma: no cover
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


def load_tokenizer_and_model(
    base_model_id: str,
    adapter_path: str,
    *,
    trust_remote_code: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    kwargs: dict = {"trust_remote_code": trust_remote_code, "torch_dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(base_model_id, **kwargs)
    adapter_path = Path(adapter_path).resolve()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter não encontrado: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    if not torch.cuda.is_available():
        model = model.to(torch.device("cpu"))
    return tokenizer, model


def build_text_generation_pipeline(
    tokenizer,
    model,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> Pipeline:
    # `transformers.pipeline` valida classes suportadas; `PeftModelForCausalLM`
    # pode ser rejeitado mesmo quando o modelo base (ex.: Qwen/Falcon) é suportado.
    # Aqui extraímos o backbone já instrumentado com LoRA para passar na validação.
    model_for_pipeline = model
    if isinstance(model, PeftModel):
        model_for_pipeline = model.base_model.model

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-generation",
        model=model_for_pipeline,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1e-6,
        top_p=0.9,
        return_full_text=False,
        device=device,
    )


def build_langchain_llm(
    pipe: Pipeline,
    *,
    model_id: str | None = None,
) -> HuggingFacePipeline:
    return HuggingFacePipeline(
        pipeline=pipe,
        model_id=model_id or "local-medpt-peft",
    )
