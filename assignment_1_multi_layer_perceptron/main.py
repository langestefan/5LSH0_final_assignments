from torchvision import datasets
import torch
import torchvision
import numpy as np
import network
from math import floor
import time


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
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_d = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size_train_s, shuffle=True)

    test_d = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transform),
        batch_size=batch_size_test_s, shuffle=True)

    return train_d, test_d


def train():
    """
    Train the network for 1 epoch.
    """
    print('Execute training')
    one_hot_label = np.zeros(10, dtype=np.uint8)
    vt_w_out_old, vt_b_out_old, vt_w_hidden_old, vt_b_hidden_old = 0, 0, 0, 0

    # loop over all mini-batches
    for batch_id, (mini_batch, label) in enumerate(train_data):

        sum_grad_w_hidden = np.zeros((network.input_dim, network.hidden1_dim))
        sum_grad_w_output = np.zeros((network.hidden1_dim, network.output_dim))
        sum_grad_b_hidden = np.zeros(network.hidden1_dim)
        sum_grad_b_output = np.zeros(network.output_dim)
        loss = 0

        # loop over all samples in mini-batch
        for sample_id, sample in enumerate(mini_batch):
            # Flatten input, create 748-dimension input vector
            flat_sample = (np.array(sample)).reshape((network.input_dim, 1))

            # Forward pass one sample to network
            one_hot_label[label[sample_id]] = 1  # we require one-hot encoding for our input data
            lossr, result = network.forward_pass(flat_sample, one_hot_label)
            loss += lossr

            # start backward pass
            w_out, b_out, w_hidden, b_hidden = network.backward_pass(one_hot_label)
            sum_grad_w_output += w_out
            sum_grad_b_output += b_out
            sum_grad_w_hidden += w_hidden
            sum_grad_b_hidden += b_hidden

            # clear label at the end of each sample
            one_hot_label[:] = 0  # clear variable

        # average gradient over mini-batch
        grad_w_output = sum_grad_w_output / batch_size_train
        grad_b_output = sum_grad_b_output / batch_size_train
        grad_w_hidden = sum_grad_w_hidden / batch_size_train
        grad_b_hidden = sum_grad_b_hidden / batch_size_train

        # add momentum gradient descent
        # see https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
        # vt = momentum * vt_old + LR * gradient
        vt_w_out = momentum * vt_w_out_old + LR * grad_w_output
        vt_b_out = momentum * vt_b_out_old + LR * grad_b_output
        vt_w_hidden = momentum * vt_w_hidden_old + LR * grad_w_hidden
        vt_b_hidden = momentum * vt_b_hidden_old + LR * grad_b_hidden

        # store vt_old for use in next iteration
        vt_w_out_old, vt_b_out_old, vt_w_hidden_old, vt_b_hidden_old = vt_w_out, vt_b_out, vt_w_hidden, vt_b_hidden

        # update network parameters
        network.weights_hidden1_output = network.weights_hidden1_output - vt_w_out
        network.bias_output = network.bias_output - vt_b_out
        network.weights_input_hidden1 = network.weights_input_hidden1 - vt_w_hidden
        network.bias_hidden1 = network.bias_hidden1 - vt_b_hidden

        # if batch_id % 100 == 0:
        #     print('Batch {0}: Loss: {1}'.format(batch_id, loss / batch_size_train))


def test():
    """
    Run test batches on trained network
    :return: N/A
    """
    print('Execute testing')
    one_hot_label = np.zeros(10, dtype=np.uint8)
    correct_n = 0
    total_n = 0

    for batch_id, (mini_batch, label) in enumerate(test_data):

        for sample_id, sample in enumerate(mini_batch):
            # Flatten input, create 748, input vector
            flat_sample = (np.array(sample)).reshape((network.input_dim, 1))

            # Forward pass one sample to network
            one_hot_label[label[sample_id]] = 1  # we require one-hot encoding for our input data
            lossr, result = network.forward_pass(flat_sample, one_hot_label)

            # print('result {}'.format(result))
            # print('label {}'.format(one_hot_label))

            # check if sample was correctly classified
            if (result == one_hot_label).all():
                correct_n += 1

            total_n += 1
            one_hot_label[:] = 0

    t_accuracy = (correct_n / total_n) * 100
    return t_accuracy


if __name__ == '__main__':
    start_time = time.time()
    network = network.Network()

    # train settings
    n_train_samples = 60000
    batch_size_test = 1000

    batch_size_train = 16
    LR = 0.001
    n_epochs = 20
    momentum = 0.9

    # network settings
    network.input_dim = 28 * 28  # 784 pixels
    network.hidden1_dim = 128
    network.output_dim = 10

    # set to True if using ReLU activation, set to False if using sigmoid
    network.relu = True

    # init network
    n_mini_batch = floor(n_train_samples / batch_size_train)
    network.init_neurons()
    network.init_weights()

    # import data
    train_data, test_data = import_data(batch_size_train, batch_size_test)

    # train
    for epoch in range(n_epochs):
        print('~~~~ {} seconds ~~~~'.format(round(time.time() - start_time, 0)))
        print('Epoch: {}'.format(epoch))
        train()

        # calculate test accuracy
        test_accuracy = test()
        print('Test accuracy: {}%'.format(test_accuracy))

    # calculate train accuracy
    test_data = train_data
    train_accuracy = test()
    print('Train accuracy: {}%'.format(train_accuracy))
