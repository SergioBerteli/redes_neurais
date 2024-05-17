from datasets import logica_and, autor_cient_m_f, cientista_e_compositores
from rede import RedeNeural




if __name__ == '__main__':
    qtd_neuronios_entrada = 2
    qtd_neuronios_saida = 2
    taxa_aprendizado = 1
    RNA = RedeNeural(qtd_neuronios_entrada, qtd_neuronios_saida, taxa_aprendizado)
    RNA.treinar_rede(0, autor_cient_m_f)
    print("Rede treinada!")
    while 1:
        rep_bin = list(map(int, list(input("insira a representação binária: "))))
        print(f"O resultado é {RNA.testes(rep_bin)}")