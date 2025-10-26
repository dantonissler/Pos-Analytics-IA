import numpy as np

def encode(x): return np.random.normal(0, 1, size=(2,))
def decode(z): return z[0] + z[1]

entrada = np.random.rand(5)
z = encode(entrada)
saida = decode(z)
print("Entrada:", entrada)
print("Espaço latente:", z)
print("Saída reconstruída:", saida)
