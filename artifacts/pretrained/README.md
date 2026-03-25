# Referência ao modelo pré-treinado (base)

Esta pasta **não** contém os pesos completos do LLM base, por serem **grandes** (GB) e sujeitos à **licença** do provedor no Hugging Face.

O que fica **versionado** aqui é apenas a **evidência documental**:

- `pretrained_manifest.json` — atualizado pelo `train_medpt.py` a cada execução, registrando **qual** checkpoint pré-treinado foi usado (ID no Hub ou caminho local).

Os tensores do modelo base continuam no **Hugging Face Hub** (download em tempo de execução via `from_pretrained`) ou no caminho local que você indicar em `--model_name_or_path`.

O resultado do **fine-tuning** (adapters LoRA + tokenizer + evidências) fica em `artifacts/finetuned/run_<timestamp>/`.
