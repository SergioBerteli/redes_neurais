from numpy import zeros
def funcao_ativacao(x) -> int:
    """
    Como função de ativação foi escolhida a função degrau
    """
    return 1 if x > 0 else 0

class Neuronio:
    def __init__(self, n_ligacoes: int, x_i: float = 0) -> None:
        self.__x_i = x_i
        self.__sinapses = zeros((n_ligacoes,), dtype=int)
    
    def setX_i(self, x_i: float) -> None:
        self.__x_i = x_i

    def getX_i(self) -> float:
        return self.__x_i 
        



class RedeNeural:
    def __init__(self, i_neuronios, o_neuronios) -> None:
        __camada_entrada = []
        __camada_entrada.append(Neuronio(o_neuronios, 1)) # Bias
        for _ in range(i_neuronios):
            __camada_entrada.append(Neuronio(o_neuronios)) #instanciando a camada de entrada
        __camada_saida = []
        for _ in range(o_neuronios):
            __camada_saida.append(Neuronio(0)) # instanciando a camada de saida