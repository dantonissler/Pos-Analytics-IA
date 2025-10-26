# Dois mini-cases: (A) triagem de currículos com ML; (B) manutenção preditiva com dados simulados.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# (A) RH: triagem de currículos
cv_texts = [
    "Python, FastAPI, PostgreSQL, 5 anos de experiência em backend",
    "Excel avançado, PowerBI, rotinas financeiras",
    "React, TypeScript, UI/UX, design system",
    "Java, Spring, microsserviços, AWS",
    "Atendimento ao cliente, vendas, metas e CRM",
    "SageMaker, MLops, scikit-learn, pipelines de IA",
]
labels = [1,0,0,1,0,1]
Xtr, Xte, ytr, yte = train_test_split(cv_texts, labels, test_size=0.33, random_state=42)
vec = TfidfVectorizer()
Xtrv, Xtev = vec.fit_transform(Xtr), vec.transform(Xte)
clf = LogisticRegression(max_iter=1000).fit(Xtrv, ytr)
pred = clf.predict(Xtev)
print("[RH] Triagem de CVs — classificação (1=match):")
print(classification_report(yte, pred))

# (B) Manutenção preditiva
rng = np.random.default_rng(42)
n = 500
vibr = rng.normal(0.5, 0.15, n)
temp = rng.normal(60, 5, n)
corr = rng.normal(10, 1.5, n)
y = ((vibr>0.65) & (temp>63)) | (corr>12)
X = pd.DataFrame({"vibracao":vibr,"temperatura":temp,"corrente":corr})
Xtr, Xte, ytr, yte = train_test_split(X, y.astype(int), test_size=0.25, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
print("\n[Manutenção] Acurácia:", rf.score(Xte, yte))
print("[Manutenção] Exemplo de prob. de falha nas 5 primeiras linhas de teste:\n", rf.predict_proba(Xte)[:5,1])
