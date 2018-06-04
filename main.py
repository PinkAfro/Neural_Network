import numpy as np
from math import *

class Neural_Network():
    def __init__(self, input_size,output_size):
        self.network_blueprint = [input_size, output_size]
        self.network = []
        self.phi = np.vectorize(self.sigmoid)

    def add_layer(self, size):
        self.network_blueprint.insert(-1, size)

    def generate_network(self):
        #Generates value and weight matrix placeholders (zeroed arrays)
        for layer_index, size in enumerate(self.network_blueprint):
            if layer_index != len(self.network_blueprint)-1:
                next_size = self.network_blueprint[layer_index+1]
                values = np.ones(shape=(1,size+1)) # size +1 for bias
                weights = np.random.randn(size+1,next_size)*np.sqrt(2/next_size)
                self.network.append([values,weights])

    def calculate(self, input_values):
        for layer in self.network:
            values = layer[0]
            #Set values to the output of the last layer if not first layer
            try:
                values[:] = output[:,-1] #Ensure bias does not affect next layer
            except:
                values[:,:-1] = input_values

            weights = layer[1]
            output = self.phi(values@weights)
            print(output)


    def parse_activation(self, phi):
        self.phi = np.vectorize(phi)

    def sigmoid(self,x):
        return 1/(1+exp(-x))



def main():
    x = np.array([0,4,2,5,6])
    network = Neural_Network(input_size=5, output_size=1)
    network.add_layer(size=3)
    network.generate_network()
    network.calculate(x)
    return


if __name__ == "__main__":
    main()