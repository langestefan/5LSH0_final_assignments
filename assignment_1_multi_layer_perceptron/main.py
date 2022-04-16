from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import network
from math import floor
import time


def import_data(train_b_size, test_b_size, shuffle=True):
    """
    This function imports the MNIST dataset. We use pytorch to load the dataset as tensors.
    :param shuffle: shuffle test/train data
    :param train_b_size: Batch size for training
    :param test_b_size: Batch size for testing
    :return: Train/test data objects
    """
    print('--- Importing data ---')
    testing_data = datasets.MNIST('./data', train=False, download=True, transform=ToTensor())
    n_test_samp = len(testing_data)  # total number of test samples
    # n_testb = int(np.floor(len(testing_data) / test_b_size))  # number of train batches

    training_data = datasets.MNIST('./data', train=True, download=True, transform=ToTensor())
    n_train_samp = len(training_data)  # total number of train samples
    # n_trainb = int(np.floor(len(training_data) / train_b_size))  # number of test batches

    # data batch arrays. Each index is a batch
    test_imgs_batched = []
    test_labels_batched = []
    train_imgs_batched = []
    train_labels_batched = []

    mean = 0.1307
    std = 0.3081

    # shuffle our indices
    if shuffle:
        rng = np.random.default_rng()
        test_ind = rng.permutation(n_test_samp)
        train_ind = rng.permutation(n_train_samp)

    # construct test set
    for i in range(0, n_test_samp, test_b_size):
        ind = test_ind[i:i + test_b_size]
        batch_x = [(testing_data[i][0].numpy() - mean) / std for i in ind]  # construct our test batch
        batch_y = [testing_data[i][1] for i in ind]  # do the same for corresponding labels
        test_imgs_batched.append(batch_x)
        test_labels_batched.append(batch_y)

    # construct train set
    for i in range(0, n_train_samp, train_b_size):
        ind = train_ind[i:i + train_b_size]
        batch_x = [(training_data[i][0].numpy() - mean) / std for i in ind]   # construct our train batch
        batch_y = [training_data[i][1] for i in ind]  # do the same for corresponding labels
        train_imgs_batched.append(batch_x)
        train_labels_batched.append(batch_y)

    # zip labels with data
    test_data_batched = zip(test_imgs_batched, test_labels_batched)
    train_data_batched = zip(train_imgs_batched, train_labels_batched)

    # return train_d, test_d
    return train_data_batched, test_data_batched


def train(traindata):
    """
    Train the network for 1 epoch.
    """
    print('--- Execute training ---')
    one_hot_label = np.zeros(10, dtype=np.uint8)
    vt_w_out_old, vt_b_out_old, vt_w_hidden_old, vt_b_hidden_old = 0, 0, 0, 0

    # loop over all mini-batches
    for batch_id, (mini_batch, label) in enumerate(traindata):

        sum_grad_w_hidden = np.zeros((network.input_dim, network.hidden_dim))
        sum_grad_w_output = np.zeros((network.hidden_dim, network.output_dim))
        sum_grad_b_hidden = np.zeros(network.hidden_dim)
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
        # vt = momentum * vt_old + LR * gradient
        vt_w_out = momentum * vt_w_out_old + LR * grad_w_output
        vt_b_out = momentum * vt_b_out_old + LR * grad_b_output
        vt_w_hidden = momentum * vt_w_hidden_old + LR * grad_w_hidden
        vt_b_hidden = momentum * vt_b_hidden_old + LR * grad_b_hidden

        # store vt_old for use in next iteration
        vt_w_out_old, vt_b_out_old, vt_w_hidden_old, vt_b_hidden_old = vt_w_out, vt_b_out, vt_w_hidden, vt_b_hidden

        # update network parameters
        network.weights_hidden_output = network.weights_hidden_output - vt_w_out
        network.bias_output = network.bias_output - vt_b_out
        network.weights_input_hidden = network.weights_input_hidden - vt_w_hidden
        network.bias_hidden = network.bias_hidden - vt_b_hidden

        # if batch_id % 100 == 0:
        #     print('Batch {0}: Loss: {1}'.format(batch_id, loss / batch_size_train))


def test(input_test_data):
    """
    Run test batches on trained network
    :return: Test accuracy [0-1]
    """
    print('--- Execute testing ---')
    one_hot_label = np.zeros(10, dtype=np.uint8)
    correct_n = 0
    total_n = 0

    for batch_id, (mini_batch, label) in enumerate(input_test_data):

        for sample_id, sample in enumerate(mini_batch):
            # Flatten input, create 748, input vector
            flat_sample = (np.array(sample)).reshape((network.input_dim, 1))

            # Forward pass one sample to network
            one_hot_label[label[sample_id]] = 1  # we require one-hot encoding for our input data
            lossr, result = network.forward_pass(flat_sample, one_hot_label)

            # check if sample was correctly classified
            if (result == one_hot_label).all():
                correct_n += 1

            total_n += 1
            one_hot_label[:] = 0
            
    # print('batch_id at end: ', batch_id)
    if total_n != 0:
        return (correct_n / total_n) * 100
    else:
        print('Warning, total_n should not be 0')
        return 0


if __name__ == '__main__':
    start_time = time.time()
    network = network.Network()

    # train settings
    n_train_samples = 60000
    batch_size_test = 1000

    batch_size_train = 8
    LR = 0.0001
    n_epochs = 30
    momentum = 0.99

    # network settings
    network.input_dim = 28 * 28  # 784 pixels
    network.hidden_dim = 256
    network.output_dim = 10

    # set to True if using ReLU activation, set to False if using sigmoid
    network.relu = True

    # init network
    n_mini_batch = floor(n_train_samples / batch_size_train)
    network.init_neurons()
    network.init_weights()

    # import data
    train_data, test_data = import_data(batch_size_train, batch_size_test, shuffle=True)

    # make sure data list is iterable
    train_data_iter = tuple(train_data)
    test_data_iter = tuple(test_data)

    # train
    for epoch in range(n_epochs):
        print('--- Epoch: {} - {} seconds ---'.format(epoch, round(time.time() - start_time, 0)))

        # calculate test accuracy
        test_accuracy = test(test_data_iter)
        print('Test accuracy: {}%'.format(test_accuracy))

        # 1 training epoch
        train(train_data_iter)

        # calculate train accuracy
        # test_data = train_data
        # train_accuracy = test(test_data)
        # print('Train accuracy: {}%'.format(train_accuracy))
