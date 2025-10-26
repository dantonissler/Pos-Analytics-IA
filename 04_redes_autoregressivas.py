import numpy as np

dados = [1, 2, 3, 4, 5]
for i in range(5):
    proximo = sum(dados[-3:]) / 3
    dados.append(proximo)
print("Sequência gerada:", dados)
