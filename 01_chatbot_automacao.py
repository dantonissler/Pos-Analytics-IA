# Chatbot simples com classificação de intenções (TF-IDF + LinearSVC) e fallback semântico.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

faq_base = [
    ("qual o horário de atendimento?", "horario"),
    ("vocês atendem aos sábados?", "horario"),
    ("preciso de 2ª via do boleto", "boleto"),
    ("como emitir 2ª via do boleto?", "boleto"),
    ("quero falar com humano", "humano"),
    ("reclamação sobre entrega atrasada", "reclamacao"),
    ("minha entrega está atrasada", "reclamacao"),
]
X, y = zip(*faq_base)

clf = Pipeline([("tfidf", TfidfVectorizer()), ("svm", LinearSVC())]).fit(X, y)
tfidf = clf.named_steps["tfidf"]
X_vec = tfidf.fit_transform(X)  # vetoriza base para fallback semântico

RESPOSTAS = {
    "horario": "Nosso atendimento é de seg a sex, das 8h às 18h (horário de Brasília).",
    "boleto": "Para 2ª via do boleto: acesse app > Financeiro > Boletos > Gerar 2ª via.",
    "humano": "Encaminhei você para um atendente humano. Aguarde um instante, por favor.",
    "reclamacao": "Sinto muito pelo transtorno. Já abri um ticket prioritário; você receberá atualização por e-mail.",
    "_fallback": "Não entendi perfeitamente. Você pode reformular? Posso ajudar com: horário, 2ª via de boleto, reclamações."
}

def responder(msg: str) -> str:
    # 1) tenta classificar intenção
    intent = clf.predict([msg])[0]
    # 2) valida confiança via similaridade de cosseno com a base (fallback se muito baixa)
    vec_user = tfidf.transform([msg])
    sim = cosine_similarity(vec_user, X_vec).max()
    if sim < 0.35:  # limiar simples
        return RESPOSTAS["_fallback"]
    return RESPOSTAS.get(intent, RESPOSTAS["_fallback"])

if __name__ == "__main__":
    testes = [
        "qual é o horário de funcionamento?",
        "segunda via do meu boleto",
        "minha entrega ainda não chegou",
        "quero conversar com alguém",
        "vocês têm estacionamento para clientes?"
    ]
    for t in testes:
        print(f"Usuário: {t}\nBot   : {responder(t)}\n")
