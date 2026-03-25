# Artefatos locais (evidências e separação pré-treinado × fine-tuning)

| Caminho | Conteúdo |
|---------|----------|
| `pretrained/` | **Referência** ao modelo base pré-treinado (`pretrained_manifest.json` + README). **Não** armazena os pesos completos do LLM base (GB). |
| `finetuned/<run>/` | **Resultado do fine-tuning**: adapters PEFT, tokenizer copiado, `evidence.json` (metadados do run). Pesos grandes costumam ser ignorados pelo `.gitignore`. |
| `logs/audit.jsonl` | **Auditoria do treino** (`train_medpt.py`): `TRAIN_START`, `TRAIN_END`, `DATA_GOVERNANCE_APPLIED`, etc. |
| `logs/assistant_audit.jsonl` | **Auditoria do assistente** (Etapa 3): `ASSISTANT_RUN_START`, `ASSISTANT_SAFETY_FLAGS`, `ASSISTANT_RUN_END` — ver `python -m assistant.run_assistant`. |

Gerado por `train_medpt.py` e por `assistant/run_assistant.py` (padrão `--artifacts-root artifacts`).
