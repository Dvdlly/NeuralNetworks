import numpy as np
import random

class QuadraticCost(object):
    @staticmethod
    def cost(a,y):
        return 0.5 * np.linalg.norm(a-y) ** 2

    def delta_L(a,y,z):
        return (a-y)*sigmoid_derivative(z)

class CrossEntropyCost(object):
    @staticmethod
    def cost(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta_L(a,y,z):
        return a-y

class FullyConnectedNN(object):
    def __init__(self, sizes,cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost_function = cost
        self.init_weights()

    def init_weights(self):
        self.biases = [np.zeros(bias,1) for bias in self.sizes]
        self.weights = [np.random.randn(i,j)/np.sqrt(j) for i,j in zip(sizes[:-1],sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self,training_data,learning_rate,batch_size,epochs,test_data=None):
        for i in range(epochs):
            print("Epoch " + str(i))
            training_data = random.shuffle(training_data)

            for k in range(0,len(training_data),batch_size):
                mini_batch = training_data[k:k+batch_size]
                self.update_mini_batch(mini_batch,learning_rate)


    def update_mini_batch(self, mini_batch, learning_rate):
        biases = [np.zeros(b.shape) for b in self.biases]
        weights = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            updated_b, updated_w = self.backpropagation(x,y)
            biases = [b+upd_b for b,upd_b in zip(biases,updated_b)]
            weights = [w+upd_w for w,upd_w in zip(weights,updated_w)]

        self.weights = [weight - (learning_rate/len(mini_batch))*upd_w \
        for weight,upd_w in zip(self.weights, weights)]

        self.biases = [bias - (learning_rate/len(mini_batch))*upd_b \
        for bias,upd_b in zip(self.biases,biases)]

    def backpropagation(self,x,y):
        updated_weights = [np.zeros(w.shape) for w in self.weights]
        updated_biases = [np.zeros(b.shape) for b in self.biases]
        a_values = [x]
        z_values = []
        a = x
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i],a)+self.biases[i]
            a = self.sigmoid(z)
            a_values.append(a)
            z_values.append(z)
        delta_L = (self.cost).delta_L(a_values[-1],y,z_values[-1])

        for j in range(len(updated_weights)-1,-1,-1):
            updated_biases[j] = delta_L
            updated_weights[j] = np.dot(delta_L, a_values[j].transpose())
            delta_L = np.dot(self.weights[j].transpose(),delta_L) * \
            self.sigmoid_derivative(a_values[j])
        return updated_biases,updated_weights

    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_derivative(self,z):
        return self.sigmoid(z) * (1-self.sigmoid(z))
