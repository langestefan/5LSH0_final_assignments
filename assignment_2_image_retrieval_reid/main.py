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

    # loop over all mini-batches
    for batch_id, (mini_batch, label) in enumerate(train_data):
        # we input our data in N = <bs_train> samples at a time
        output = network(mini_batch)
        # compute loss
        loss = lossf(output, label)
        # make sure old gradients are set to 0 before the next step
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        optimizer.step()

        if batch_id % 100 == 0:
            loss = loss.item()
            current = batch_id * len(mini_batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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
    i = 0

    # with .no_grad() we make sure pytorch does not perform backprogation/gradient calculations
    with torch.no_grad():
        for data, labels in test_data:
            network.twod_feature_vectors[:, 2] = labels  # store for later use
            print('Running test batch: {0}'.format(i))
            i += 1

            output = network(data)
            loss += lossf(output, labels).item()

            # check if digit was correctly classified
            pred_label = torch.max(output, 1)[1].data.squeeze()

            for idx, label in enumerate(labels):
                total_n[label] += 1

                if pred_label[idx] == label:
                    correct_n[label] += 1

        loss /= num_batches

        # compute correct/total for every digit
        correct_frac = correct_n / total_n

        print('Fraction correct for each label [0-9]: {}'.format(np.round(correct_frac, 3)))
        print('Total test loss: {}'.format(np.round(loss, 5)))


# network class
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.twod_feature_vectors = np.zeros((bs_test, 3))
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
        # if we are testing (not training) we store 2D vector result for use later
        if not self.training:
            print('x: {}'.format(np.shape(x)))
            self.twod_feature_vectors[:, 0:2] = x

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
    fig, ax = plt.subplots()
    for (points, color, digit) in zip(all_points, colors, range(10)):
        # print('points: {0}, color: {1}, digit: {2}'.format(points, color, digit))
        ax.scatter([item[0] for item in points],
                   [item[1] for item in points],
                   color=color, label='digit{}'.format(digit))
        ax.grid(True)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    # network settings
    bs_test = 1000

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
    print(network)

    # optimizer for backpropagation
    optimizer = optim.SGD(network.parameters(), lr=LR, momentum=momentum)
    # print(optimizer)

    start_time = time.time()  # to keep track of runtime

    # nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.
    lossf = nn.CrossEntropyLoss()

    # test to see performance without any training
    # test()

    # train
    for epoch in range(n_epochs):
        print('--- Epoch: {0} - {1} seconds ---'.format(epoch, round(time.time() - start_time, 0)))
        # train()  # 1 training epoch
        # test()  # test updated model

        # The output of the 2-D FC layer will act as the feature vector of the input image
        # (i.e. the embedding of the input image). The previously mentioned ranking of the gallery set
        # by the similarity to the given query is measured using this feature vector.

        # print 2D feature vector of last <bs_test> test samples
        print('twod_feature_vector: {0}'.format(network.twod_feature_vectors))

    # plot 20 samples of each digit
    # plot the same 20 samples such that you can see how the points move
    pts = np.random.rand(10, 20, 2)  # n_digits=10, n_points=20, (x,y) columns=2
    plot(pts)
