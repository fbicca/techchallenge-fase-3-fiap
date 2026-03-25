# Tech Challenge — Fine-tuning MedPT + Assistente LangChain

Projeto acadêmico em etapas: **(1)** fine-tuning supervisionado (LoRA) com **MedPT**; **(2)** assistente médico com **LangChain** + SQLite; **(3)** **segurança, auditoria e explainability**.

---

## Estrutura rápida

| Caminho | Descrição |
|---------|-----------|
| `train_medpt.py` | Fine-tuning (SFT + LoRA), governança de dados, `artifacts/` |
| `evaluate_finetune.py` | Métricas NLL/PPL (e opcional ROUGE) base vs adapter |
| `data_governance.py` | Pré-processamento, anonimização, curadoria |
| `assistant/` | Modelo + LangChain, SQLite demo, segurança e auditoria (Etapa 2–3) |
| `requirements.txt` / `requirements-langchain.txt` | Dependências |

---

## Passo a passo de execução (do zero)

### 0) Pré-requisitos

- Python 3.10+ (recomendado 3.11/3.12).
- Ambiente virtual ativo.
- Espaço em disco para o modelo base + artefatos.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -r requirements-langchain.txt
```

### 1) Modelo pré-treinado (base)

Este projeto usa por padrão `Qwen/Qwen2.5-1.5B-Instruct` (foco em latência melhor para chatbot).

- O modelo base é baixado em tempo de execução via `from_pretrained`.
- A referência do checkpoint usado é registrada em `artifacts/pretrained/pretrained_manifest.json`.
- Se necessário, autentique no Hugging Face:

```bash
huggingface-cli login
```

### 2) Fine-tuning (Etapa 1)

Treino completo (salva adapter + tokenizer + evidências):

```bash
python train_medpt.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --run_name qwen-medpt
```

Saída esperada:

- `artifacts/finetuned/qwen-medpt/` com adapter PEFT + tokenizer.
- `artifacts/finetuned/qwen-medpt/evidence.json`.
- `artifacts/logs/audit.jsonl`.

**Nota técnica (LoRA por arquitetura):**
- O script `train_medpt.py` seleciona automaticamente `target_modules` de LoRA conforme o `model.config.model_type`.
- Exemplo: em Falcon usa `query_key_value`/`dense*`; em Qwen/Llama-like usa `q_proj`/`k_proj`/`v_proj`/`o_proj` e MLP.
- Isso evita o erro `Target modules ... not found in the base model` ao trocar de modelo base.

### 3) Smoke test rápido de treino (opcional)

Para validar pipeline sem custo alto:

```bash
python train_medpt.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --max_samples 200 \
  --num_train_epochs 0.1 \
  --run_name smoke-qwen
```

### 3b) Avaliação do fine-tuning (métricas)

O script `evaluate_finetune.py` compara **modelo base** vs **base + adapter** na mesma fatia do MedPT, com a mesma governança e formatação do treino:

- **NLL / perplexidade** (teacher forcing): perda média ao prever o texto completo (incluindo a resposta de referência). **Quanto menor, melhor** o encaixe à fatia avaliada.
- **`delta_mean_nll` negativo** e **`relative_ppl_ratio` inferior a 1** indicam que o fine-tuning **melhorou** relativamente ao base nessa fatia.

Use uma fatia **disjunta** da usada no treino (ex.: treino nos primeiros 5000 exemplos, avaliação a seguir):

```bash
pip install rouge-score   # opcional, só para --with-rouge

# Use sempre o nome da flag antes do valor (ex.: --model-name-or-path ...).
# Uma linha evita erro de continuação no shell:
python evaluate_finetune.py --model-name-or-path Qwen/Qwen2.5-1.5B-Instruct --adapter-path artifacts/finetuned/qwen-fast --eval-start 5000 --eval-max-samples 500 --output-json artifacts/eval/metrics.json
```

Multilinha (cada linha termina com `\` e **sem** espaço depois da barra):

```bash
python evaluate_finetune.py \
  --model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path artifacts/finetuned/qwen-fast \
  --eval-start 5000 \
  --eval-max-samples 500 \
  --output-json artifacts/eval/metrics.json
```

- `--skip-base`: avalia só o fine-tuned (mais rápido se já tiver métricas do base guardadas).
- `--with-rouge`: gera respostas e calcula ROUGE-L F1 vs referência (mais lento; usa até `--rouge-max-samples` linhas).

### 4) Execução do assistente (Etapa 2)

Use o adapter gerado no passo anterior:

```bash
python -m assistant.run_assistant \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path artifacts/finetuned/qwen-medpt \
  --question "Quais alergias constam e o que aparece nos registros recentes?"
```

Se o nome do run for outro, veja com:

```bash
ls artifacts/finetuned
```

> Importante: não use `<` e `>` no bash (ex.: `<run>`), pois isso vira redirecionamento de shell.

---

## Etapa 1 (resumo)

- Dataset: [AKCIT/MedPT](https://huggingface.co/datasets/AKCIT/MedPT).
- Modelo base: `Qwen/Qwen2.5-1.5B-Instruct` (Hub público, sem gated da Meta).
- Treino: `python train_medpt.py --run_name qwen-medpt` gera adapter em `artifacts/finetuned/qwen-medpt/`.

---

## Etapa 2 (resumo)

- `pip install -r requirements-langchain.txt`
- `python -m assistant.run_assistant --adapter-path artifacts/finetuned/qwen-medpt --question "..."`

> Substitua `qwen-medpt` pelo nome real do seu run (veja com `ls artifacts/finetuned`). Não use `<` e `>` no bash.

Conformidade com o enunciado: pipeline LangChain + LLM fine-tuned; SQLite (prontuário demo); no modo `context` o resumo inclui **todos os pacientes** da base demo (sem filtro por ID).

### LangGraph (fluxos modulares)

O assistente usa **LangGraph** para orquestrar nós reutilizáveis (estado tipado, fácil de estender):

| Modo | Fluxo |
|------|--------|
| `context` | `fetch_prontuario` → `generate` → `safety_scan` |
| `sql` | `sql_agent` → `safety_scan` |

| Ficheiro | Papel |
|----------|--------|
| `assistant/langgraph_state.py` | Tipos de estado (`ContextAssistantState`, `SqlAssistantState`). |
| `assistant/langgraph_nodes.py` | Nós (carregar prontuário, gerar resposta, agente SQL, varredura de segurança). |
| `assistant/langgraph_graphs.py` | Compilação dos grafos (`compile_context_assistant_graph`, `compile_sql_assistant_graph`). |
| `assistant/prompts.py` | Montagem do prompt de contexto partilhada entre grafos e cadeia linear opcional. |
| `assistant/sql_agent_factory.py` | Construção do agente SQL (evita import circular com `chains.py`). |

`python -m assistant.run_assistant` invoca estes grafos diretamente; `assistant/chains.py` pode expor a mesma lógica via `RunnableLambda` para integração LangChain. Instalação: `pip install -r requirements-langchain.txt` (inclui `langgraph`).

---

## Etapa 3 — Segurança, validação e explainability

Este bloco atende ao item **3. Segurança e validação** do desafio.

### 1) Limites de atuação do assistente

- **Prompt de sistema reforçado** (`assistant/security.py`, constante `SYSTEM_LIMITS_AND_SAFETY`), injetado na cadeia em `assistant/chains.py`:
  - não prescrever medicamentos nem doses;
  - não substituir diagnóstico ou conduta exclusiva do médico;
  - recusar pedidos de receita/posologia e orientar validação humana;
  - não incentivar automedicação;
  - usar o contexto do prontuário quando existir.
- **Heurística de auditoria** (`scan_response_for_safety_flags`): após a geração, o texto é escaneado por padrões que *podem* indicar posologia ou linguagem de prescrição; **não bloqueia** a resposta automaticamente, mas **registra flags** no log para revisão (ex.: `possivel_mencao_posologia_numerica`).

> Em produção, somariam-se políticas de conteúdo, revisão humana e conformidade regulatória; aqui o foco é **demonstração acadêmica explícita**.

### 2) Logging para rastreamento e auditoria

- Ficheiro **append-only** em JSON Lines: **`artifacts/logs/assistant_audit.jsonl`** (configurável via `--artifacts-root`).
- Eventos típicos:
  - `ASSISTANT_RUN_START` — modo, pergunta, caminhos de modelo/adapter/base de dados;
  - `ASSISTANT_SAFETY_FLAGS` — flags heurísticas da resposta;
  - `ASSISTANT_RUN_END` — tamanho da resposta, modo.
- Implementação: `assistant/assistant_audit.py` (`append_assistant_audit`).
- **Desligar** (não recomendado para entrega): `--no-assistant-audit`.

### 3) Explainability (fontes da informação)

- **Modo texto (padrão):** após a resposta do modelo, é impresso um bloco **“Fontes e transparência”** com:
  - identificação do **modelo base** e do **caminho do adapter** (fine-tuning);
  - no modo **`context`**: ficheiro SQLite usado para montar o resumo de **todos** os prontuários da demo;
  - no modo **`sql`**: referência ao agente SQL e tabelas do demo;
  - reforço dos **limites** (não prescrever, validação humana).
- Função: `format_explainability_block` em `assistant/security.py`.
- **Modo JSON estruturado:** `--json` imprime um único objeto JSON com:
  - `answer` — texto gerado;
  - `sources` — lista de objetos (`type`, `detail`, `file` / `adapter_path` quando aplicável);
  - `safety_heuristic_flags`;
  - `limits` — resumo textual da política.

**Ocultar o rodapé markdown** (mantendo JSON/auditoria): `--no-explainability-footer`.

### Como utilizar (Etapa 3)

Na raiz do projeto, com venv ativo e dependências LangChain instaladas:

```bash
# Resposta em texto + bloco markdown de fontes/limites + auditoria em assistant_audit.jsonl
python -m assistant.run_assistant \
  --adapter-path artifacts/finetuned/smoke-qwen \
  --question "Quais alergias constam no prontuário?"

# Saída apenas JSON (útil para integração e relatórios)
python -m assistant.run_assistant --json \
  --adapter-path artifacts/finetuned/smoke-qwen \
  --question "Resuma medicações ativas."

# Sem escrita em assistant_audit.jsonl (apenas testes locais)
python -m assistant.run_assistant --no-assistant-audit \
  --adapter-path artifacts/finetuned/smoke-qwen \
  --question "..."
```

| Flag | Efeito |
|------|--------|
| `--json` | Saída estruturada com `sources` e flags de segurança. |
| `--no-explainability-footer` | Não imprime o bloco markdown de fontes (a auditoria em ficheiro continua, salvo `--no-assistant-audit`). |
| `--no-assistant-audit` | Desativa `assistant_audit.jsonl`. |
| `--artifacts-root` | Pasta raiz onde fica `logs/assistant_audit.jsonl` (padrão `artifacts`). |

### Conformidade com o enunciado (item 3)

| Requisito | Implementação |
|-----------|----------------|
| Limites para evitar sugestões impróprias (ex.: não prescrever sem validação humana) | `SYSTEM_LIMITS_AND_SAFETY` no sistema do chat + texto explícito no rodapé explainability. |
| Logging detalhado para auditoria | `artifacts/logs/assistant_audit.jsonl` com eventos por execução. |
| Explainability (indicar fontes) | Rodapé markdown e/ou `--json` com campo `sources` (SQLite + modelo/adapter). |

---

## Considerações legais e éticas

Dados do SQLite são **fictícios**. O assistente **não** substitui profissional de saúde. Respeite licenças do **MedPT**, do **Qwen** e a **LGPD** em cenários com dados reais.

---

## Licença

Código de estudo e pesquisa; adapte citações ao dataset e ao modelo base utilizados.
