import time

def automatizar(tarefas):
    for t in tarefas:
        print(f"Executando: {t}")
        time.sleep(0.5)
    print("Tarefas concluídas com eficiência e sem erros humanos!")

tarefas = ["Backup diário", "Geração de relatório", "Envio de e-mails"]
automatizar(tarefas)
