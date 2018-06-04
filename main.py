import numpy as np

class Neural_Network():
    def __init__(self, input_size,output_size):
        self.network_blueprint = [input_size, output_size]
        self.network = []

    def add_layer(self, size):
        self.network_blueprint.insert(-1, size)

    def generate_network(self):
        for layer_index, size in enumerate(self.network_blueprint):
            if layer_index != len(self.network_blueprint)-1:
                next_size = self.network_blueprint[layer_index+1]
                values = np.zeros(shape=(1,size))
                weights = np.zeros(shape=(size,next_size))
                self.network.append([values,weights])

    def parse_activation(self, phi):
        self.phi = phi


def main():
    network = Neural_Network(input_size=3, output_size=1)
    network.add_layer(size=3)
    network.generate_network()
    return


if __name__ == "__main__":
    main()