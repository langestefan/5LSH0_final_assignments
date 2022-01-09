from torchvision import datasets
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
# from re_id import query


# network class
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # to store 2d feature vector results
        self.feature_vectors_ground_truth_test = np.zeros((bs_test, 3))
        self.feature_vectors_prediction_test = np.zeros((bs_test, 3))

        self.feature_vectors_train = []  # np.zeros((0, 3))
        self.train_labels = []

        # layer conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # layer conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # layer conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=7),  # global max of each filter
            nn.Flatten(),  # flatten to create 32 dimensional input vector for fully connected layer
            nn.Linear(32, 2),  # 2D FC
        )
        self.fc2 = nn.Linear(2, 10)  # 10D FC

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Store 2D vector result for use with re-id later
        if not self.training:
            self.feature_vectors_ground_truth_test[:, 0:2] = x
            self.feature_vectors_prediction_test[:, 0:2] = x

        if self.training:
            self.feature_vectors_train.append(x.detach().numpy())
            # print('2d_vectors_train: {0}'.format(self.feature_vectors_train))
            # print('2d_vectors_train: {0}'.format(type(self.feature_vectors_train)))
            # print('2d_vectors_train: {0}'.format(np.shape(self.feature_vectors_train)))

        x = self.fc2(x)
        return x


# Plot and show
def plot(all_points):
    """
    Plot function from assignment document
    :param all_points:
    :return:
    """
    colors = matplotlib.cm.Paired(np.linspace(0, 1, len(all_points)))
    fig, ax = plt.subplots(figsize=(7, 5))
    for (points, color, digit) in zip(all_points, colors, range(10)):
        # print('points: {0}, color: {1}, digit: {2}'.format(points, color, digit))
        ax.scatter([item[0] for item in points],
                   [item[1] for item in points],
                   color=color, label='digit{}'.format(digit))
        ax.grid(True)
    ax.legend()
    plt.show()


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
        batch_size=batch_size_test_s, shuffle=False)  # shuffle to false to track points over epochs

    return train_d, test_d


def train():
    """
    This function is responsible for training the network's parameters.
    This code is losely based on the optimization tutorial from PyTorch, see:
    https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    """
    print('--- start training ---')
    network.train()
    size = len(train_data.dataset)
    network.train_labels = []

    # loop over all mini-batches
    for batch_id, (mini_batch, tlabel) in enumerate(train_data):
        network.train_labels.append(tlabel.numpy())  # save training labels
        # print('train_labels: {0}'.format(np.shape(train_labels)))
        # we input our data in N = <bs_train> samples at a time
        output = network(mini_batch)
        # compute loss
        loss = lossf(output, tlabel)
        # make sure old gradients are set to 0 before the next step
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        optimizer.step()

        if batch_id % 100 == 0:
            loss = loss.item()
            current = batch_id * len(mini_batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # save network state after each training epoch
    # torch.save(network.state_dict(), 'model.pth')
    # torch.save(optimizer.state_dict(), 'optimizer.pth')


def test():
    """
    Run the trained model on test data.
    This code is losely based on the optimization tutorial from PyTorch, see:
    https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    """
    print('--- start test ---')
    size = len(test_data.dataset)
    num_batches = len(test_data)
    loss = 0

    correct_n = np.zeros([10])  # we store correct_n for each digit individually
    total_n = np.zeros([10])

    # Make sure entire network is enabled and in test mode via .eval()
    network.eval()

    # with .no_grad() we make sure pytorch does not perform backprogation/gradient calculations
    with torch.no_grad():
        for data, labels in test_data:
            network.feature_vectors_ground_truth_test[:, 2] = labels  # store groundtruth

            output = network(data)
            loss += lossf(output, labels).item()

            # check if digit was correctly classified
            pred_label = torch.max(output, 1)[1].data.squeeze()

            network.feature_vectors_prediction_test[:, 2] = pred_label

            for idx, label in enumerate(labels):
                total_n[label] += 1

                if pred_label[idx] == label:
                    correct_n[label] += 1

        loss /= num_batches

        # compute correct/total for every digit
        correct_frac = correct_n / total_n

        print('Fraction correct for each label [0-9]: {}'.format(np.round(correct_frac, 3)))
        print('Total test loss: {}'.format(np.round(loss, 5)))


if __name__ == '__main__':
    # network settings
    bs_test = 10000  # 1000

    # hyperparameters
    LR = 0.01
    momentum = 0.9
    n_epochs = 1
    bs_train = 64

    # torch settings
    torch.backends.cudnn.enabled = False  # turn off GPU processing
    torch.manual_seed(1)  # to keep random numbers the same over executions

    # import data
    train_data, test_data = import_data(bs_train, bs_test)

    # create network object
    network = Network()
    network.load_state_dict(torch.load('model.pth'))
    print(network)

    # optimizer for backpropagation
    optimizer = optim.SGD(network.parameters(), lr=LR, momentum=momentum)
    # print(optimizer)

    start_time = time.time()  # to keep track of runtime

    # nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.
    lossf = nn.CrossEntropyLoss()

    # train
    for epoch in range(n_epochs):
        print('--- Epoch: {0} - {1} seconds ---'.format(epoch, round(time.time() - start_time, 0)))
        # to hold 2d feature vectors for TEST DATA
        sorted_features = []
        sorted_features_plot = np.zeros((10, 20, 2))

        # to hold gallery images array(60000, 3) = (x0, y0, digit)
        gallery = np.zeros((0, 2))

        # we test once without training, then we complete the epoch loop
        test()  # test updated model

        # print 2D feature vector of last N=<bs_test> test samples
        a = network.feature_vectors_ground_truth_test
        for id_x in range(10):
            sorted_features.append(a[np.where(a[:, 2] == id_x)][:, :2])  # to store the data in a convenient way
            sorted_features_plot[id_x] = a[np.where(a[:, 2] == id_x)][:20, :2]  # to hold just the data for plotting

        # plot 2D feature embedding space of test data
        plot(sorted_features_plot)

        # 1 training epoch
        train()

        # generate gallery = train set with predicted labels

        for i, (x0, y0, predlabel) in enumerate(network.feature_vectors_prediction_test):
            print(i, x0, y0, predlabel)

        # do re-id query
        # query = test set
        # gallery = train set

        # reshape self.feature_vectors_train into (60000,3) np array
        # then array[:,3] = train_labels[:]
        # self.feature_vectors_train

        # for idx, d_list in enumerate(sorted_features):
        #     print(idx)
        #     print(np.shape(d_list))
        #
        #     results = query((d_list[0], d_list[1]), )

