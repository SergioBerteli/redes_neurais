from numpy import exp, dot, zeros
from numpy.random import standard_normal
from random import shuffle

def sigmoide(x):
    return 1/(1+exp(-x))


class NeuralNetwork:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [standard_normal((i, 1)) for i in sizes[1:]]
        self.weights = [standard_normal((i, j)) for i, j in zip(sizes[:-1])]
    
    def feedfoward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoide(dot(w, a)+b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
            shuffle(training_data)
            mini_batches = [training_data[k: k+mini_batch_size]
                           for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {i}: {self.evaluate(test_data), n_test}")
            else:
                print(f"Epoch {i} complete")
    
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [zeros(b.shape) for b in self.biases]
        nabla_w = [zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

