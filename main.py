from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from math import *
from mnist import MNIST
import time
import os


class Neural_Network_Numpy():
    def __init__(self, input_size, output_size):
        self.network_blueprint = [input_size, output_size]
        self.network = []
        self.phi = np.vectorize(self.sigmoid)
        self.cost = self.quadratic
        self.output_error = self.output_error_quadratic
        self.previous_error = self.previous_error_quadratic
        self.a_list = []  # acitvation neurons where the index is the layer - 1
        self.b_list = [None]  # bias where the index is the layer -1
        self.w_list = [None]  # weights where the index is the layer - 1
        self.z_lists = []  # A list of z values from every input
        self.a_lists = []  # Contains the state of the activations from every input
        self.cost_list = []  # Cost of every input
        self.error_lists = []  # List of errors where error_lists[x] is the error from input_values_lists[x]
        self.pC_pw_list = []  # List of pC_pW for every input value for every x -> [x][layer]
        self.tmp_a_list = []

    def clear_lists(self):
        self.cost_list = []  # Cost of every input
        self.error_lists = []  # List of errors where error_lists[x] is the error from input_values_lists[x]
        self.pC_pw_list = []  # List of pC_pW for every input value for every x -> [x][layer]
        self.error_lists = []
        self.z_lists = []
        self.a_lists = []

    def add_layer(self, size):
        self.network_blueprint.insert(-1, size)

    def generate_network(self, load=None):
        # Generates value and weight matrix
        # j_neurons is the number of neurons in (L-1) layer, k_neurons is the number in the Lth layer
        if load:
            self.load_network(load)
            print('Loaded %s' % load)
        else:
            for layer_index, j_neurons in enumerate(self.network_blueprint):
                a = np.ones(shape=(j_neurons, 1))
                self.a_list.append(a)

                if layer_index != len(self.network_blueprint) - 1:
                    k_neurons = self.network_blueprint[layer_index + 1]
                    w = np.random.randn(k_neurons, j_neurons) * np.sqrt(2 / k_neurons)
                    self.w_list.append(w)

                    b = np.random.randn(k_neurons, 1)
                    self.b_list.append(b)
        self.tmp_a_list = self.a_list[:]

    def save_network(self, filename):
        names = ['_weights', '_bias', '_activations', '_network']
        parameters = {0: self.w_list, 1: self.b_list, 2: self.tmp_a_list, 3: self.network_blueprint}
        for index, name in enumerate(names):
            np.savez(filename + name, parameters[index])

    def load_network(self, filename):
        names = ['_weights', '_bias', '_activations', '_network']
        self.w_list = np.load(filename + names[0] + '.npz')['arr_0'].tolist()
        self.b_list = np.load(filename + names[1] + '.npz')['arr_0'].tolist()
        self.a_list = np.load(filename + names[2] + '.npz')['arr_0'].tolist()
        self.network_blueprint = np.load(filename + names[3] + '.npz')['arr_0'].tolist()

    def calculate(self, input_values_list):
        for input_values in input_values_list:
            self.a_list[0] = input_values
            z_list = []
            for layer in range(len(self.network_blueprint) - 1):
                a = self.a_list[layer]
                b = self.b_list[layer + 1]
                w = self.w_list[layer + 1]

                z = (w @ a) + b

                z_list.append(z)

                a_next = self.phi(z)
                self.a_list[layer + 1] = a_next

            self.a_lists.append(self.a_list[:])
            self.z_lists.append(z_list[:])

    def test(self, input_values_list, output_values_list, normalising_thingo):
        test = []
        label_thingy = []
        good_checker = 0
        self.calculate(input_values_list)
        listy = self.a_lists
        for a in listy:
            test.append(np.rint(a[-1] * normalising_thingo))

        for label in output_values_list:
            label_thingy.append(label * normalising_thingo)

        for input, output in zip(test, label_thingy):
            if input == output:
                good_checker = good_checker + 1

        print(label_thingy)
        print(test)
        print(good_checker / len(test) * 100)

    def back_propogate(self, input_values_list, output_values_list, learning_rate):
        self.cost_list = []
        if len(input_values_list) != len(output_values_list):
            print('Input and Output lists must be the same length')
            exit(1)

        self.calculate(input_values_list)
        # Calculates the error for every input
        for input_index in range(len(input_values_list)):
            a = self.a_lists[input_index][-1]
            z = self.z_lists[input_index][-1]
            y = output_values_list[input_index]

            cost = self.cost(a, y)
            self.cost_list.append(cost)

            self.output_error(a, z, y)
            self.previous_error(input_index)
            self.pC_pw(input_index)

        total_cost = self.total_cost(self.cost_list)
        print('cost: %s' % total_cost)
        self.gradient_decent(learning_rate)
        self.tmp_a_list = self.a_list
        self.clear_lists()
        return

    def gradient_decent(self, learning_rate):
        error_list, n = self.sum(self.error_lists)
        pC_pw_list, m = self.sum(self.pC_pw_list)

        if n != m:
            print('ERROR: Error list and pc_pw lists are different lengths')
            exit(1)

        b_list = [None]
        w_list = [None]
        for layer in range(len(self.network_blueprint) - 1):
            # Calculate gradient decent for bias
            b = self.b_list[layer + 1]
            error = error_list[layer + 1]
            b = b - (learning_rate / m) * error
            b_list.append(b)

            # Calculate gradient decent for weights
            w = self.w_list[layer + 1]
            pC_pw = pC_pw_list[layer]
            w = w - (learning_rate / m) * pC_pw
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
            if layer != len(self.w_list) - 1:
                a = self.a_lists[input_index][layer]
                dimensions = (1, self.network_blueprint[layer + 1])
                a = np.tile(a, dimensions).T
                error = self.error_lists[input_index][layer + 1]
                pC_pw = np.multiply(error, a)
                pC_pw_list.append(pC_pw[:])
        self.pC_pw_list.append(pC_pw_list[:])

    def previous_error_quadratic(self, input_index):
        for layer in range(len(self.w_list) - 2):
            w = self.w_list[-(layer + 1)]
            z = self.z_lists[input_index][-(layer + 2)]
            error_next = self.error_list[1]
            error = np.multiply((w.T @ error_next), (self.dsigma_dz(z)))
            self.error_list.insert(1, error)
        self.error_lists.append(self.error_list[:])

    def dsigma_dz(self, z):  # dx_dy is the derivative of x with respect to y
        return (np.exp(-z) / ((1 + np.exp(-z)) ** 2))

    def output_error_quadratic(self, a, z, y):
        pC_pa = (a - y)  # pX_pY represents the partial derivative X with respect to Y
        output_error = np.multiply(pC_pa, self.dsigma_dz(z))
        self.error_list = [None, output_error]
        return output_error

    def total_cost(self, cost_list):
        cost = sum(cost_list)  # * (1 / (2 * n))
        return cost

    def parse_activation(self, phi):
        self.phi = np.vectorize(phi)

    def parse_cost(self, cost):
        self.cost = np.vectorize(cost)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def quadratic(self, a, y):
        magnitude = np.linalg.norm(y - a)
        return (1 / 2) * (magnitude ** 2)


class Neural_Network_Tensor:
    def __init__(self, inputs=None, outputs=None, load=None):
        tf.enable_eager_execution()
        self.activation = self.sigmoid
        if load:
            self.load(load)
        elif inputs and outputs:
            self.inputs = inputs
            self.outputs = outputs
            self.layers_neurons = []
            self.w = [None]
            self.b = [None]
            self.z_layers = []
            self.a_layers = []
            self.error_list = [None]
            self.gradient_list = []


        else:
            print('ERROR: Inputs or Outputs not specified')
            exit(1)

    def clear_lists(self):
        self.z_layers = []
        self.a_layers = []
        self.error_list = [None]
        self.gradient_list = []

    def add_layer(self, neurons):
        self.layers_neurons.append(neurons)

    def generate_network(self):
        # self.a = tf.placeholder(np.float64)
        # a = np.ones(shape=(j_neurons, 1))
        self.layers_neurons.append(self.outputs)
        self.layers_neurons.insert(0, self.inputs)
        for index in range(len(self.layers_neurons) - 1):
            w = tf.truncated_normal([self.layers_neurons[index + 1], self.layers_neurons[index]], dtype=np.float64)
            b = tf.ones([self.layers_neurons[index + 1], 1], dtype=np.float64)
            self.w.append(w)
            self.b.append(b)

    def calculate(self, input_array):
        self.clear_lists()
        self.length = len(input_array)
        # init_g = tf.global_variables_initializer()
        # init_l = tf.local_variables_initializer()
        # sess = tf.Session()
        #
        # # a = self.a
        self.a_layers.append(input_array.T)  # Set the first layer to the input

        for layer_index in range(len(self.layers_neurons) - 1):  # (len(self.w)-1):
            z = tf.matmul(self.w[layer_index + 1], self.a_layers[layer_index]) + self.b[layer_index + 1]
            activation = self.sigmoid(z)

            self.a_layers.append(activation)
            self.z_layers.append(z)

        self.neural_output = activation

    def cost_gradient(self):
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()

        a = self.a_layers[0]
        error = self.error_list[1]

        for layer in range(len(self.a_layers) - 1):
            a = self.a_layers[layer]
            error = self.error_list[layer + 1]
            error = tf.transpose(error)
            cost = tf.matmul(a, error)
            self.gradient_list.append(cost)

    def gradient_decent(self, learning_rate):
        w_list = [None]
        b_list = [None]
        m = self.length
        for layer in range(len(self.w) - 1):
            error = self.error_list[layer + 1]
            error = tf.reduce_sum(error, 1, keepdims=True)

            gradient = tf.transpose(self.gradient_list[layer])

            w = self.w[layer + 1]
            b = self.b[layer + 1]

            b = b - (learning_rate / m) * error
            w = w - (learning_rate / m) * gradient

            w_list.append(w)
            b_list.append(b)

        self.w = w_list[:]
        self.b = b_list[:]
        return

    def train(self, input, output, learning_rate):
        start_calc = time.time()
        self.calculate(input)
        end_calc = time.time()

        start_out_error = time.time()
        output_error = self.output_error_quadratic(output)
        end_out_error = time.time()

        start_back = time.time()
        self.backpropogate_error(output_error)
        end_back = time.time()

        start_cost = time.time()
        self.cost_gradient()
        end_cost = time.time()

        start_decent = time.time()
        self.gradient_decent(learning_rate)
        end_decent = time.time()

        # start_cost_final = time.time()
        # cost = self.quadratic(self.neural_output, output)
        # end_cost_final = time.time()

        # print('Calculate: %s\nOutput_error: %s\nBackpropogate: %s\nCost_gradient: %s\nGradient_Decent: %s\nCost: %s\nTotal: %s' % (
        #     end_calc - start_calc, end_out_error - start_out_error, end_back - start_back, end_cost - start_cost,
        #     end_decent - start_decent, end_cost_final-start_cost_final, end_cost_final-start_calc))


    def backpropogate_error(self, output_error):
        self.error_list.append(output_error)

        for layer in range(len(self.layers_neurons) - 2):
            w = tf.transpose(self.w[-(layer + 1)])
            z = self.z_layers[-(layer + 2)]
            error_next = self.error_list[1]
            w_dot_error = tf.matmul(w, error_next)
            error = np.multiply((w_dot_error), (self.dsigma_dz(z)))
            self.error_list.insert(1, error)

    def quadratic(self, neural_output, output_correct):
        # print(output_correct.shape, neural_output.T.shape)
        magnitude = output_correct - neural_output.flatten()
        # cost = (1/self.length)*(np.linalg.norm((1 / 2) * (magnitude ** 2)))
        cost = np.average((1 / 2) * (magnitude ** 2))
        return cost

    def output_error_quadratic(self, output):
        a = self.a_layers[-1]
        z = self.a_layers[-1]

        pC_pa = (a - output)  # pX_pY represents the partial derivative X with respect to Y
        output_error = np.multiply(pC_pa, self.dsigma_dz(z))

        return output_error

    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    def dsigma_dz(self, z):  # dx_dy is the derivative of x with respect to y
        z = np.clip(z, np.finfo(np.float64).min, np.finfo(np.float64).max)
        return (np.exp(-z) / ((1 + np.exp(-z)) ** 2))

    def save(self, filename):
        names = ['_weights', '_bias', '_network']
        parameters = {0: self.w, 1: self.b,2: self.layers_neurons}
        for index, name in enumerate(names):
            tmp = []
            for layer in parameters[index]:
                if layer is not None:
                    tmp.append(np.array(layer))
                else:
                    tmp.append(layer)
            np.savez(filename + name, tmp)
    #
    def load(self, filename):
        names = ['_weights', '_bias', '_network']
        self.w = np.load(filename + names[0] + '.npz')['arr_0'].tolist()
        self.b = np.load(filename + names[1] + '.npz')['arr_0'].tolist()
        self.layers_neurons = np.load(filename + names[2] + '.npz')['arr_0'].tolist()

    def training_loop(self, inputs, labels, batch_size, epochs, learning_rate, save=False):
        for epoch in range(epochs):
            total_cost = 0
            iterations = (int(ceil(len(inputs) / batch_size)))
            for i in range(iterations):
                try:
                    self.train(inputs[i * batch_size:(i * batch_size) + batch_size],
                                      labels[i * batch_size:(i * batch_size) + batch_size], learning_rate=learning_rate)
                except():
                    self.train(inputs[i * batch_size:],
                                      labels[i * batch_size:], learning_rate=learning_rate)
            print(epoch)
        if save:
            self.save(save)


def normalise(data):  #############DODGY METHOD (Works only if lower bound is 0)
    max_value = 0
    for _ in data:
        try:
            if max(_) > max_value:
                max_value = max(_)

        except:
            if _ > max_value:
                max_value = _

    return data / max_value, max_value


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # t_network = Neural_Network_Tensor(3,1)
    # t_network.add_layer(3)
    # t_network.generate_network()
    # input_data = np.asarray([[1,6,4,3,2],[7,7,5,4,3],[9,5,6,7,5]])#np.asarray([[0,3,5,6,7],[1,8,3,9,8],[9,7,6,8,6]])
    # output = np.asarray([[1],[7],[9]]).T

    # input_data, normaliser_output = normalise(input_data)
    # output, normaliser_output = normalise(output)
    #
    # print(input_data.shape, output.shape)
    # print(type(input_data),type(output))

    mndata = MNIST()
    images, labels = mndata.load_testing()
    labels = np.asarray(labels, dtype=np.float64)
    images = np.asarray(images, dtype=np.float64)
    images, normaliser_images = normalise(images)
    labels, normaliser_labels = normalise(labels)



    print(images.shape)
    print(labels.shape)

    t_network = Neural_Network_Tensor(inputs=784, outputs=1)
    t_network.add_layer(500)
    t_network.add_layer(200)
    t_network.add_layer(3)
    # t_network.generate_network()
    t_network.load('tf_test')
    t_network.training_loop(images, labels, batch_size=1000, epochs=10000, learning_rate=0.2, save='Digit_Recognition')


    #Test_Network
    #2481
    #5893
    #
    #
    # t_network.calculate(images)
    # output = t_network.neural_output.flatten()
    # output = np.round(output*normaliser_labels)
    # labels = labels * normaliser_labels
    # check = output-labels
    # print((check == 0).sum())

    #
    # for iterations in range(10):
    #     print('-------------------Iteration: %s -------------------------' % iterations)
    #     for i in range(6):
    #         t_network.train(images[], labels, learning_rate=0.5)

    # for iterations in range(10):
    #     print('-------------------Iteration: %s -------------------------' % iterations)
    #     for i in range(6):
    #         t_network.train(images[10000*i:(10000*i)+10000], labels[10000*i:(10000*i)+10000], learning_rate = 0.5)

    # tf.executing_eagerly()
    # mndata = MNIST()
    # test_images, test_labels = mndata.load_testing()
    # test_labels = test_labels.tolist()
    # test_input = []
    # test_output = []
    # for image in test_images:
    #     test_input.append((np.asarray(image)/255).reshape(784,1))
    # for label in test_labels:
    #     test_output.append(np.asarray((label)/9).reshape(1,1))
    #

    #
    # network = Neural_Network(input_size=784, output_size=1)
    # network.add_layer(800)
    # network.generate_network()
    #
    # for iterations in range(1):
    #     print('-------------------Iteration: %s -------------------------' % iterations)
    #     for i in range(100):
    #         network.back_propogate(input[600*i:(600*i)+600], output[600*i:(600*i)+600], learning_rate = 0.5)
    #
    # network.save_network('MNIST_2')
    # network.test(test_input, test_output, 9)

    return


if __name__ == "__main__":
    main()
