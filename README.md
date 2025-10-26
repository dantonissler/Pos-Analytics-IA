# 🤖 Inteligência Artificial e Automação de Processos

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Concluído-success.svg)

---

## 🧭 Sobre o Projeto

Este repositório reúne **conceitos fundamentais de Inteligência Artificial aplicada à automação de processos**, baseando-se em questões teóricas e exemplos práticos que ilustram o uso real de IA em:

- Atendimento automatizado (chatbots)
- Processamento de linguagem natural (NLP)
- Aprendizado de máquina supervisionado e não supervisionado
- Modelos generativos (VAEs, redes autoregressivas)
- Plataformas de Big Data e IA (Databricks)
- Previsão de cenários e análise de dados

Cada arquivo `.py` ou `.md` representa um **módulo conceitual** vinculado a uma questão do conjunto de estudos.

---

## 🧩 Estrutura do Repositório

| Módulo | Tema | Conceitos-chave |
|---------|------|-----------------|
| **01_chatbot_automacao.py** | Automação de atendimento com IA | NLP, chatbots, aprendizado supervisionado |
| **02_analise_sentimentos.py** | Análise de sentimentos | Extração de emoções, NLP, polaridade |
| **03_automacao_tarefas.py** | Automação de tarefas repetitivas | RPA, scripts, macros, eficiência operacional |
| **04_redes_autoregressivas.py** | Redes Neurais Autoregressivas | Modelos generativos, previsão passo a passo |
| **05_ia_empresarial.py** | Aplicações de IA em RH e manutenção | Aprendizado de máquina, NLP, IoT |
| **06_autoencoder_variacional.py** | Autoencoders Variacionais (VAE) | Geração de conteúdo, espaço latente |
| **07_databricks_intro.md** | Databricks e Apache Spark | Big Data, nuvem, processamento distribuído |
| **08_previsao_cenarios.py** | Previsão de dados e análise de cenários | Regressão, simulação, visualização |
| **09_algoritmos_ml_basicos.py** | Conceitos de ML: clustering, classificação e anomalias | KMeans, DecisionTree, IsolationForest |
| **10_automatizacao_ia_empresas.py** | Automação inteligente empresarial | IA + produtividade + redução de erros |

---

## 🧠 Exemplos Práticos

### 🔹 01 — Chatbot Automatizado com IA

```python
from transformers import pipeline

chatbot = pipeline("text-generation", model="distilgpt2")
pergunta = "Olá, como posso te ajudar?"
resposta = chatbot(f"Usuário: {pergunta} IA:", max_length=50, num_return_sequences=1)
print(resposta[0]['generated_text'])
