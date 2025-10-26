# Exemplo de “orquestrador” que integra:
# (1) classificação automática de chamados, (2) regra de priorização, (3) envio para fila.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import queue, time

# Base mínima (texto -> categoria)
dados = [
    ("sistema fora do ar desde cedo", "incidente"),
    ("erro 500 no endpoint de login", "incidente"),
    ("solicito criação de acesso", "solicitacao"),
    ("gostaria de aumentar meu limite", "solicitacao"),
    ("atualizar dados cadastrais", "solicitacao"),
    ("queda de performance em produção", "incidente"),
]
X, y = zip(*dados)
pipe = Pipeline([("tfidf", TfidfVectorizer()), ("lr", LogisticRegression(max_iter=1000))]).fit(X, y)

# fila simulada
fila_incidentes = queue.Queue()
fila_solicitacoes = queue.Queue()

def priorizar(texto: str) -> dict:
    categoria = pipe.predict([texto])[0]
    proba = max(pipe.predict_proba([texto])[0])
    prioridade = "alta" if (categoria=="incidente" and proba>0.6) else "normal"
    return {"texto":texto, "categoria":categoria, "prioridade":prioridade, "confianca":proba}

def enviar_para_fila(item: dict):
    (fila_incidentes if item["categoria"]=="incidente" else fila_solicitacoes).put(item)

def processar_lote(chamados: list[str]):
    for c in chamados:
        item = priorizar(c)
        enviar_para_fila(item)
    print(f"[OK] Lote processado. Incidentes: {fila_incidentes.qsize()} | Solicitações: {fila_solicitacoes.qsize()}")

if __name__ == "__main__":
    chamados = [
        "app travando para vários usuários",
        "preciso criar usuário novo",
        "endpoint de pagamentos retornando erro 503",
        "alterar meu endereço",
    ]
    processar_lote(chamados)
    while not fila_incidentes.empty():
        it = fila_incidentes.get(); print("[INCIDENTE]", it)
    while not fila_solicitacoes.empty():
        it = fila_solicitacoes.get(); print("[SOLICITAÇÃO]", it)
    print("-> Resultado: menos triagem manual, resposta mais rápida e padronizada (redução de erros).")
