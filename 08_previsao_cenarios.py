# Regressão + simulação de cenários (Monte Carlo)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

rng = np.random.default_rng(42)
n = 400
marketing = rng.normal(50, 10, n)
preco = rng.normal(100, 5, n)
vendas = 2.5*marketing + 1.2*(150-preco) + rng.normal(0, 15, n)
df = pd.DataFrame({"marketing":marketing,"preco":preco,"vendas":vendas})
Xtr, Xte, ytr, yte = train_test_split(df[["marketing","preco"]], df["vendas"], test_size=0.25, random_state=42)
model = LinearRegression().fit(Xtr, ytr)
pred = model.predict(Xte)
print("MAE:", mean_absolute_error(yte, pred), " | R2:", r2_score(yte, pred))
print("Coeficientes:", dict(zip(["marketing","preco"], model.coef_)), "Intercept:", model.intercept_)

def simular_cenarios(n_sims=1000, mk_mean=60, mk_sd=8, p_mean=98, p_sd=4):
    mk = rng.normal(mk_mean, mk_sd, n_sims)
    p  = rng.normal(p_mean, p_sd, n_sims)
    X  = pd.DataFrame({"marketing":mk,"preco":p})
    yhat = model.predict(X)
    return yhat

y_sims = simular_cenarios()
print("Cenário: vendas previstas (p50, p90):", np.percentile(y_sims, [50,90]))

for nome, (mk,p) in {
    "Conservador": (50, 102),
    "Base": (60, 100),
    "Agressivo": (75, 96)
}.items():
    y = model.predict(pd.DataFrame({"marketing":[mk],"preco":[p]}))[0]
    print(f"{nome}: marketing={mk}k, preço={p} => vendas previstas={y:.1f}")
