import numpy as np

import activation
from activation import cross_entropy


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

        # gradients
        # to keep track of the sum of the gradients
        self.sum_grad_w_hidden1 = []
        self.sum_grad_w_output = []
        self.sum_grad_b_hidden1 = []
        self.sum_grad_b_output = []

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
        # self.bias_hidden1 = np.random.uniform(-1, 1, self.hidden1_dim)
        # self.bias_output = np.random.uniform(-1, 1, self.output_dim)

        # input
        self.pre_hidden1 = np.zeros(self.hidden1_dim)
        self.pre_output = np.zeros(self.output_dim)

        # gradients
        self.sum_grad_w_hidden1 = np.zeros((self.input_dim, self.hidden1_dim))
        self.sum_grad_w_output = np.zeros((self.hidden1_dim, self.output_dim))
        self.sum_grad_b_hidden1 = np.zeros(self.hidden1_dim)
        self.sum_grad_b_output = np.zeros(self.output_dim)

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
        np.random.seed(5)

        # xavier weight initialization w ~ U(-1/sqrt(n), 1/sqrt(n))
        w_h1 = 1/np.sqrt(self.input_dim)
        w_o = 1/np.sqrt(self.hidden1_dim)

        # random (xavier) uniform initialization for weights
        self.weights_input_hidden1 = np.random.uniform(-w_h1, w_h1, (self.input_dim, self.hidden1_dim))
        self.weights_hidden1_output = np.random.uniform(-w_o, w_o, (self.hidden1_dim, self.output_dim))

        print('Weights vector input-->hidden1: {0}'.format(self.weights_input_hidden1.shape))
        print('Weights vector hidden1-->output: {0}'.format(self.weights_hidden1_output.shape))

    def forward_pass(self, input_data, one_hot_label):
        """
        Updates all neurons for a single sample and computes the new output
        :param one_hot_label:
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

        # input --> hidden layer 1 (ReLU)
        # output = input_vector * weight_matrix  + bias_vector
        # Y = XW + B
        self.pre_hidden1 = np.matmul(self.input, self.weights_input_hidden1) + self.bias_hidden1
        self.hidden1_activation = activation.relu(self.pre_hidden1)

        # hidden layer 1 --> output (SoftMax)
        self.pre_output = np.matmul(self.hidden1_activation, self.weights_hidden1_output) + self.bias_output
        self.output_activation = activation.softmax(self.pre_output)

        # after forward pass we return the loss using Cross Entropy loss function
        loss = cross_entropy(self.output_activation, one_hot_label)

        # non-maximum supression on output vector
        result = np.zeros_like(self.output_activation, dtype=np.uint8)
        result[self.output_activation.argmax(0)] = 1

        return loss, result

    def backward_pass(self, label):
        """
        Compute just derivatives for a single backwards pass. This function will NOT update the weights!
        :param label: One-hot encoded label vector of length 10. Example: [0 0 0 0 0 0 1 0 0 0] = 7
        :return: N/A?
        """
        # gradient from error to the input of the output (softmax) node
        gradient_output = activation.softmax_bw(self.output_activation, label)
        # print('gradient: {}'.format(gradient_output))

        # 1. compute gradient for weights in output layer, should give (40,10) matrix
        # gradient_wij = h_j * (o_i - y_i) = vect_H^T * vect_gradient_output
        grad_w_output = np.matmul(np.transpose([self.hidden1_activation]), [gradient_output])

        # 2. compute gradient for biases in output layer
        # gradient_b = 1 * (o_i - y_i) = o_i - y_i
        grad_b_output = gradient_output

        # 3. compute gradient for weights in hidden layer, should give (784,40) matrix
        # we need to sum the gradient over the previous layer since there are now multiple paths
        # gradient_w_hidden = sum(gradient_output) * relu_bw * input_i

        # must be 40, vector for gradient of h1 hidden output
        gradient_h1_s = np.sum(gradient_output * self.weights_hidden1_output, axis=1)

        # must be 40, bw propagation for ReLU
        relu_bw = activation.relu_bw(self.pre_hidden1)

        # this is the gradient from the error function to the input of the hidden layer
        gradient_e_to_h1_in = gradient_h1_s * relu_bw

        # matrix product of gradient_e_to_h1_in with
        grad_w_hidden = np.matmul(np.transpose([self.input]), [gradient_e_to_h1_in])

        # 4. compute gradient for biases in hidden layer
        grad_b_hidden = gradient_e_to_h1_in

        return grad_w_output, grad_b_output, grad_w_hidden, grad_b_hidden
