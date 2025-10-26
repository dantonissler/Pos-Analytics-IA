from transformers import pipeline

analisar = pipeline("sentiment-analysis")
texto = "Estou muito satisfeito com o serviço!"
resultado = analisar(texto)
print(resultado)
