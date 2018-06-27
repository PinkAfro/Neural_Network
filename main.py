import numpy as np
from math import *


class Neural_Network():
    def __init__(self, input_size, output_size):
        self.network_blueprint = [input_size, output_size]
        self.network = []
        self.phi = np.vectorize(self.sigmoid)
        self.cost = self.quadratic
        self.output_error = self.output_error_quadratic
        self.previous_error = self.previous_error_quadratic


    def add_layer(self, size):
        self.network_blueprint.insert(-1, size)

    def generate_network(self):
        # Generates value and weight matrix
        # j_neurons is the number of neurons in (L-1) layer, k_neurons is the number in the Lth layer
        self.a_list = [] #acitvation neurons where the index is the layer - 1
        self.b_list = [None] # bias where the index is the layer -1
        self.w_list = [None] # weights where the index is the layer - 1
        self.z_lists = [] # A list of z values from every input
        self.a_lists = [] # Contains the state of the activations from every input
        self.cost_list = [] # Cost of every input
        self.error_lists = [] # List of errors where error_lists[x] is the error from input_values_lists[x]
        self.pC_pw_list = [] # List of pC_pW for every input value

        for layer_index, j_neurons in enumerate(self.network_blueprint):
            a = np.ones(shape=(j_neurons, 1))
            self.a_list.append(a)

            if layer_index != len(self.network_blueprint) - 1:
                k_neurons = self.network_blueprint[layer_index + 1]
                w = np.random.randn(k_neurons, j_neurons) * np.sqrt(2 / k_neurons)
                self.w_list.append(w)

                b = np.random.randn(k_neurons, 1)
                self.b_list.append(b)


    def calculate(self, input_values_list):
        for input_values in input_values_list:
            self.a_list[0] = input_values
            z_list = []
            for layer in range(len(self.network_blueprint)-1):
                a = self.a_list[layer]
                b = self.b_list[layer + 1]
                w = self.w_list[layer + 1]

                z = (w @ a) + b
                z_list.append(z)

                a_next = self.phi(z)
                self.a_list[layer + 1] = a_next

            self.a_lists.append(self.a_list[:])
            self.z_lists.append(z_list[:])




    def back_propogate(self, input_values_list, output_values_list):
        if len(input_values_list) != len(output_values_list):
            print('Input and Output lists must be the same length')
            exit(1)

        self.calculate(input_values_list)
        #Calculates the error for every input
        for input_index in range(len(input_values_list)):
            a = self.a_lists[input_index][-1]
            z = self.z_lists[input_index][-1]
            y = output_values_list[input_index]

            cost = self.cost(a,y)
            self.cost_list.append(cost)

            self.output_error(a,z,y)
            self.previous_error(input_index)
            self.pC_pw(input_index)


        total_cost = self.total_cost(self.cost_list)
        print('cost: %s' % total_cost)
        return


    def pC_pw(self, input_index):
        for layer in range(len(self.w_list)):
            if layer !=  len(self.w_list)-1:
                a = self.a_lists[input_index][layer]
                dimensions = (1,self.network_blueprint[layer+1])
                a = np.tile(a, dimensions).T
                error = self.error_lists[input_index][layer+1]
                pC_pw = np.multiply(error,a)
                self.pC_pw_list.append(pC_pw[:])

    def previous_error_quadratic(self,input_index):
        for layer in range(len(self.w_list)-2):
            w = self.w_list[-(layer+1)]
            z = self.z_lists[input_index][-(layer+2)]
            error_next = self.error_list[1]
            error = np.multiply((w.T @ error_next), (self.dsigma_dz(z)))
            self.error_list.insert(1,error)
        self.error_lists.append(self.error_list[:])

    def dsigma_dz(self, z): # dx_dy is the derivative of x with respect to y
        return (np.exp(-z)/((1+np.exp(-z))**2))

    def output_error_quadratic(self, a,z,y):
        pC_pa = (a-y) # pX_pY represents the partial derivative X with respect to Y
        output_error = np.multiply(pC_pa,self.dsigma_dz(z))
        self.error_list = [None,output_error]
        return output_error

    def total_cost(self, cost_list):
        cost = sum(cost_list) #* (1 / (2 * n))
        return cost

    def parse_activation(self, phi):
        self.phi = np.vectorize(phi)

    def parse_cost(self, cost):
        self.cost = np.vectorize(cost)

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def quadratic(self, a, y):
        magnitude = np.linalg.norm(y - a)
        return (1 / 2) * (magnitude ** 2)


def main():
    x = [np.array([[1], [4], [2], [5], [6]]), np.array([[0], [4], [3], [5], [4]])]
    y = [np.array([[1]]), np.array([[5]])]
    network = Neural_Network(input_size=5, output_size=1)
    network.add_layer(size=3)
    network.generate_network()
    network.back_propogate(x, y)
    return


if __name__ == "__main__":
    main()
