from random import shuffle
from numpy import zeros, array, stack
from datasets.cientista_compositor import cientista_e_compositores
def funcao_ativacao(x) -> int:
    """
    Como função de ativação foi escolhida a função degrau
    """
    return 1 if x > 0 else 0

class Neuronio:
    def __init__(self, n_ligacoes: int, x_i: float = 0) -> None:
        self.__x_i = x_i # Define o valor do neurônio
        self.__sinapses = zeros((n_ligacoes,), dtype=int) # define os pesos sinápticos que começarão em 0
    
    def ajusta_pesos(self, taxa_aprendizado, erro):
        novos_pesos = []
        for index, sinapse in enumerate(self.getSinapses()):
            novos_pesos = sinapse + erro[index] * taxa_aprendizado * self.getX_i
        
        self.setSinapses(novos_pesos)

    def setX_i(self, x_i: float) -> None:
        self.__x_i = x_i

    def getX_i(self) -> float:
        return self.__x_i 
    
    def getSinapses(self):
        return self.__sinapses
    
    def setSinapses(self, sinapses):
        self.__sinapses = sinapses



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
        erro_conj_array = []
        while erro_conj_max < erro_conj_per:
            shuffle(dataset)
            for sample in dataset:
                self.insere_entrada(sample[0])
                self.gera_saida()
                # backpropagation
                erro = array(sample[1] - self.getCamadaSaida()) # calculo de erro para backpropagation TODO
                
    
    def gera_saida(self):
        entradas = self.getCamadaEntrada()
        matriz_pesos =  stack([neuronio.getSinapses() for neuronio in entradas], axis=-1)
        vetor_valores_n = array([neuronio.getX_i() for neuronio in self.getCamadaEntrada()])
        saida = matriz_pesos @ vetor_valores_n
        self.insere_saida(saida)
        
    def ajusta_peso_neuronios(self, erro):
        pass # TODO 
    
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

if __name__ == "__main__":
    RNA = RedeNeural(2, 1)
    RNA.treinar_rede(0, cientista_e_compositores)