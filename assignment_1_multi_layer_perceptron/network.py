import numpy as np

import activation
from activation import relu, sigmoid


class Network:
    def __init__(self):
        # input layer
        self.input_dim = 1
        self.input = []

        # hidden layer 1
        self.hidden1_dim = 1
        self.hidden1_activation = []
        self.weights_input_hidden1 = []
        self.pre_hidden1 = []
        self.bias_hidden1 = []

        # output layer
        self.output_dim = 1
        self.output_activation = []
        self.weights_hidden1_output = []
        self.pre_output = []
        self.bias_output = []

    def init_neurons(self):
        """
        Create neuron arrays
        :return: N/A
        """
        self.input = np.zeros(self.input_dim)

        # activation
        self.hidden1_activation = np.zeros(self.hidden1_dim)
        self.output_activation = np.zeros(self.output_dim)

        # bias
        self.bias_hidden1 = np.zeros(self.hidden1_dim)
        self.bias_output = np.zeros(self.output_dim)

        # input
        self.pre_hidden1 = np.zeros(self.hidden1_dim)
        self.pre_output = np.zeros(self.output_dim)

        print('input: {0}'.format(self.input.shape))
        print('hidden1_activation: {0}'.format(self.hidden1_activation.shape))
        print('bias_hidden1: {0}'.format(self.bias_hidden1.shape))
        print('output_activation: {0}'.format(self.output_activation.shape))
        print('bias_output: {0}'.format(self.bias_output.shape))

    def init_weights(self):
        """
        Create weight arrays
        :return: N/A
        """
        # to keep same weights next time program is run
        # np.random.seed(5)
        # random uniform initialization for weights
        self.weights_input_hidden1 = np.random.uniform(-1, 1, (self.input_dim, self.hidden1_dim))
        self.weights_hidden1_output = np.random.uniform(-1, 1, (self.hidden1_dim, self.output_dim))
        print('Weights vector input-->hidden1: {0}'.format(self.weights_input_hidden1.shape))
        print('Weights vector hidden1-->output: {0}'.format(self.weights_hidden1_output.shape))

    def forward_pass(self, input_data):
        """
        Updates all neurons for a single sample and computes the new output
        :param input_data: Input array of new samples
        :return: N/A
        """
        assert np.size(input_data) == self.input_dim, "Data input dimension should equal NN input layer dimension"
        self.input = input_data[:, 0]

        # forward pass:
        # 1. Input 'neurons' activation = input value
        # 2. Calculate hidden layer 1 neuron activation = dotproduct of input*dot*weights_input_hidden1
        # 3. Softmax for the output layer
        # 4. Cross entropy for the loss layer

        # input --> hidden layer 1
        for node_id in range(self.hidden1_dim):
            # node input = dotproduct of weights with input + bias
            self.pre_hidden1[node_id] = np.dot(self.input, self.weights_input_hidden1[:, node_id]) + \
                                        self.bias_hidden1[node_id]
            # node output = activationF(input)
            node_activ = activation.relu(self.pre_hidden1[node_id])
            self.hidden1_activation[node_id] = node_activ

        # hidden layer 1 --> output
        for node_id in range(self.output_dim):
            # node input = dotproduct of weights with input + bias
            self.pre_output[node_id] = np.dot(self.hidden1_activation, self.weights_hidden1_output[:, node_id]) + \
                                       self.bias_output[node_id]

        # apply softmax to output layer
        print('Input: {0}'.format(self.pre_output))
        self.output_activation = activation.softmax(self.pre_output)
        print('Output: {0}'.format(self.output_activation))

        # after forward pass is done we return the loss

    def backward_pass(self):
        """
        Update weights
        :return:
        """
        print(1)
