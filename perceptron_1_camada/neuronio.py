from numpy import zeros, atleast_1d

class Neuronio:
    def __init__(self, n_ligacoes: int, x_i: float = 0) -> None:
        self.__x_i = x_i # Define o valor do neurônio
        self.__sinapses = zeros((n_ligacoes,), dtype=int) # define os pesos sinápticos que começarão em 0
    
    def ajusta_pesos(self, taxa_aprendizado, erro):
        novos_pesos = []
        for index, sinapse in enumerate(self.getSinapses()):
            novos_pesos.append(sinapse + erro[index] * taxa_aprendizado * self.getX_i())
        
        self.setSinapses(novos_pesos)

    def setX_i(self, x_i: float) -> None:
        self.__x_i = x_i

    def getX_i(self) -> float:
        return self.__x_i 
    
    def getSinapses(self):
        return self.__sinapses
    
    def setSinapses(self, sinapses):
        self.__sinapses = atleast_1d(sinapses)