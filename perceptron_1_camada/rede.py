from neuronio import Neuronio
from fn_ativacao import degrau
from random import shuffle
from numpy import array, stack, atleast_1d, mean, absolute 

class RedeNeural:
    def __init__(self, i_neuronios, o_neuronios, taxa_aprendizado) -> None:
        self.__n = taxa_aprendizado
        self.__camada_entrada = []
        self.__camada_entrada.append(Neuronio(o_neuronios, 1)) # Bias
        for _ in range(i_neuronios):
            self.__camada_entrada.append(Neuronio(o_neuronios)) #instanciando a camada de entrada
        self.__camada_saida = []
        for _ in range(o_neuronios):
            self.__camada_saida.append(Neuronio(0)) # instanciando a camada de saida
        self.__n_ligacoes_entrada = o_neuronios
    
    def treinar_rede(self, erro_conj_max, dataset):
        erro_conj_per = 100
        while erro_conj_max < erro_conj_per:
            erro_conj_array = []
            shuffle(dataset)
            for sample in dataset:
                self.insere_entrada(sample[0])
                self.gerar_saida()
                # backpropagation
                erro = array(array(sample[1]) - array([neuronio.getX_i() for neuronio in self.getCamadaSaida()])) # calculo de erro para backpropagation TODO
                self.ajusta_peso_neuronios(erro)
                erro_conj_array.append(mean(absolute(erro)))
            erro_conj_per = mean(erro_conj_array) * 100
                
    
    def gerar_saida(self):
        entradas = self.getCamadaEntrada()
        matriz_pesos =  stack([neuronio.getSinapses() for neuronio in entradas], axis=-1)
        vetor_valores_n = array([neuronio.getX_i() for neuronio in self.getCamadaEntrada()])
        saida = array(matriz_pesos @ vetor_valores_n)
        saida = atleast_1d(saida) #garante que a saida terá pelo menos 1 dimensão
        saida = array(list(map(degrau, saida)))
        self.insere_saida(saida)
        
    def ajusta_peso_neuronios(self, erro):
        for neuronio in self.getCamadaEntrada():
            neuronio.ajusta_pesos(self.getTaxaAprendizado(), erro)
    
    def getCamadaEntrada(self):
        return self.__camada_entrada

    def getCamadaSaida(self):
        return self.__camada_saida
    
    def getNLigacoesEntrada(self):
        return self.__n_ligacoes_entrada
    
    def getTaxaAprendizado(self):
        return self.__n

    def insere_entrada(self, data):
        for index, neuronio in enumerate(self.getCamadaEntrada()):
            if index != 0: # pulando o bias:
                neuronio.setX_i(data[index - 1])

    def insere_saida(self, data):
        for index, neuronio in enumerate(self.getCamadaSaida()):
            neuronio.setX_i(data[index])

    def setTaxaAprendizado(self, taxa_nova):
        self.__n = taxa_nova
    
    def getValoresSaida(self):
        saida_vals = []
        for neuronio in self.getCamadaSaida():
            saida_vals.append(neuronio.getX_i())
        return saida_vals

    def testes(self, inp: list):
        self.insere_entrada(inp)
        self.gerar_saida()
        return self.getValoresSaida()