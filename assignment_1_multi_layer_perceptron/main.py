from torchvision import datasets
import torch
import torchvision
import numpy as np
from activation import cross_entropy
import network
import matplotlib.pyplot as plt
from math import floor


# Some sources:
# https://towardsdatascience.com/batch-mini-batch-stochastic-gradient-descent-7a62ecba642a
# https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/
# https://sgugger.github.io/a-simple-neural-net-in-numpy.html
# https://stats.stackexchange.com/a/306710
# https://www.youtube.com/watch?v=znqbtL0fRA0&ab_channel=MLDawn


def import_data(batch_size_train_s, batch_size_test_s):
    """
    This function imports the MNIST dataset. This is the only function that makes use of pytorch, as per assignment
    requirements: "You are allowed to load the MNIST dataset in any way of your choice."
    :param batch_size_train_s: Batch size for training
    :param batch_size_test_s: Batch size for testing
    :return: Train/test data objects
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_d = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size_train_s, shuffle=True)

    test_d = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transform),
        batch_size=batch_size_test_s, shuffle=True)

    return train_d, test_d


def train():
    """
    Traing the network.
    :return: N/A
    """
    # we get 60000/64 = 937 minibatches with 60000%64 = 32 samples left
    one_hot_label = np.zeros(10, dtype=np.uint8)
    for batch_id, (mini_batch, label) in enumerate(train_data):

        sum_grad_w_hidden1 = np.zeros((network.input_dim, network.hidden1_dim))
        sum_grad_w_output = np.zeros((network.hidden1_dim, network.output_dim))
        sum_grad_b_hidden1 = np.zeros(network.hidden1_dim)
        sum_grad_b_output = np.zeros(network.output_dim)

        for sample_id, sample in enumerate(mini_batch):
            # Flatten input, create 748, input vector
            flat_sample = (np.array(sample)).reshape((network.input_dim, 1))

            # Forward pass one sample to network
            network.forward_pass(flat_sample)

            # after forward pass we return the loss using Cross Entropy loss function
            one_hot_label[label[sample_id]] = 1  # we require one-hot encoding for our input data
            loss = cross_entropy(network.output_activation, one_hot_label)

            # start backward pass
            sum_grad_w_output += network.backward_pass(one_hot_label)

            # print('sum_grad_w_output: {}'.format(sum_grad_w_output))

            # clear label at the end of each sample
            one_hot_label[:] = 0  # clear variable

        # when we exit the for loop and have treated all samples in our minibatch, we average the update to our weights
        # and update the final weights by this average
        grad_w_output = sum_grad_w_output / batch_size_train

        # update network
        network.weights_hidden1_output = learning_rate * network.weights_hidden1_output * grad_w_output

        print('Batch {0}: Loss: {1}'.format(batch_id, loss))


if __name__ == '__main__':
    network = network.Network()

    # train settings
    n_train_samples = 60000
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    n_mini_batch = floor(n_train_samples / batch_size_train)
    n_epochs = 1

    # network settings
    network.input_dim = 28 * 28  # 784 pixels
    network.hidden1_dim = 40
    network.output_dim = 10

    # init network
    network.init_neurons()
    network.init_weights()

    # import data
    train_data, test_data = import_data(batch_size_train, batch_size_test)

    # start training
    for epoch in range(n_epochs):
        train()
