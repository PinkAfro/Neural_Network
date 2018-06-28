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

    def clear_lists(self):
        self.cost_list = [] # Cost of every input
        self.error_lists = [] # List of errors where error_lists[x] is the error from input_values_lists[x]
        self.pC_pw_list = [] # List of pC_pW for every input value for every x -> [x][layer]
        self.error_lists = []
        self.z_lists = []
        self.a_lists = []

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
        self.pC_pw_list = [] # List of pC_pW for every input value for every x -> [x][layer]

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
            print(a_next)



    def back_propogate(self, input_values_list, output_values_list):
        self.cost_list = []
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
        self.gradient_decent()
        self.clear_lists()
        return

    def gradient_decent(self):
        error_list, n = self.sum(self.error_lists)
        pC_pw_list, m = self.sum(self.pC_pw_list)

        if n != m:
            print('ERROR: Error list and pc_pw lists are different lengths')
            exit(1)

        b_list = [None]
        w_list = [None]
        for layer in range(len(self.network_blueprint)-1):
            #Calculate gradient decent for bias
            learning_rate = 0.3 ################################FIGURE OUT WHAT TO MAKE THIS
            b = self.b_list[layer+1]
            error = error_list[layer+1]
            b = b - (learning_rate/m) * error
            b_list.append(b)

            #Calculate gradient decent for weights
            w = self.w_list[layer+1]
            pC_pw = pC_pw_list[layer]
            w = w - (learning_rate/m) * pC_pw
            w_list.append(w)

        self.b_list = b_list[:]
        self.w_list = w_list[:]
        # print(error)


    def sum(self, list):
        tmp_list = []
        for layer in range(len(list[0])):
            tmp = None
            for input in range(len(list)):
                if tmp is not None:
                    tmp = tmp + list[input][layer]
                else:
                    tmp = list[input][layer]
            layer_sum = tmp
            tmp_list.append(layer_sum)
        return tmp_list, len(list)

    def pC_pw(self, input_index):
        pC_pw_list = []
        for layer in range(len(self.w_list)):
            if layer !=  len(self.w_list)-1:
                a = self.a_lists[input_index][layer]
                dimensions = (1,self.network_blueprint[layer+1])
                a = np.tile(a, dimensions).T
                error = self.error_lists[input_index][layer+1]
                pC_pw = np.multiply(error,a)
                pC_pw_list.append(pC_pw[:])
        self.pC_pw_list.append(pC_pw_list[:])

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
    x = [np.array([[0.1], [0.4], [0.2], [0.5], [0.6]]), np.array([[0.0], [0.4], [0.3], [0.5], [0.4]])]
    y = [np.array([[0.1]]), np.array([[0.5]])]
    network = Neural_Network(input_size=5, output_size=1)
    network.add_layer(size=3)
    network.generate_network()
    for i in range(100000):
        network.back_propogate(x, y)
    return


if __name__ == "__main__":
    main()
