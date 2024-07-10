from numpy import randn, exp, dot

def sigmoide(x):
    return 1/(1+exp(-x))


class NeuralNetwork:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [randn(i, 1) for i in sizes[1:]]
        self.weights = [randn(i, j) for i, j in zip(sizes[:-1])]
    
    def feedfoward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoide(dot(w, a)+b)
        return a