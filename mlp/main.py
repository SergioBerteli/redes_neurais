from numpy import zeros
from random import randint 

class RedeNeural:
    def __init__(self, n_entrada, n_saida, n_neu_cam_ocul, n_cam_ocul) -> None:
        self.__n_entrada = zeros((n_entrada,), dtype=int)
        self.__n_saida = zeros((n_saida,), dtype=int)
        self.__n_cam_ocultas = zeros((n_cam_ocul, n_neu_cam_ocul), dtype=int)
        sinapses = list()
        # pesos sinapticos na camda de entrada
        sin_cam = [randint(-10, 10)/100 for _ in range(n_neu_cam_ocul)]
        while sum(sin_cam)!=0:
            sin_cam = [randint(-10, 10)/100 for _ in range(n_neu_cam_ocul)]
        print(sin_cam)
        

if __name__ == "__main__":
    rn = RedeNeural(2,1, 2, 1)
