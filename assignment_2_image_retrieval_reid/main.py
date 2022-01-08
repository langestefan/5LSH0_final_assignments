from torchvision import datasets
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
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
    network.train()
    lossf = nn.CrossEntropyLoss()

    # loop over all mini-batches
    for batch_id, (mini_batch, label) in enumerate(train_data):
        optimizer.zero_grad()

        # we input our data N = <batch_size_train> samples at a time
        output = network(mini_batch)
        print(np.shape(output))


# network class
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
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
            nn.Conv2d(
                in_channels=16,
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
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=7),  # global max of each filter

            # flatten
            nn.Flatten(),
            nn.Linear(32, 2),  # 2D FC
            nn.Linear(2, 10)  # 10D FC

        )

        # # 2D-FC
        # self.fc1 = nn.Linear(32, 2)  # 2D FC
        # # 10D-FC
        # self.fc2 = nn.Linear(2, 10)  # 10D FC

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # # reshape to fit FC layer
        # x = x.view(x.size(0), -1)
        # # 32 --> 2 Fully connected layer
        # x = self.fc1(x)
        # # 2 --> 10 Fully connected layer
        # x = self.fc2(x)
        print('fc2: {0}'.format(np.shape(x)))
        return x


if __name__ == '__main__':
    # network settings
    n_epochs = 1
    batch_size_train = 64
    batch_size_test = 1000
    LR = 0.01
    momentum = 0.5
    n_epochs = 5

    # torch settings
    seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)

    # import data
    train_data, test_data = import_data(batch_size_train, batch_size_test)

    # create network object
    network = Network()
    print(network)

    # optimizer for backpropagation
    optimizer = optim.SGD(network.parameters(), lr=LR, momentum=momentum)
    print(optimizer)

    start_time = time.time()

    # train
    for epoch in range(n_epochs):
        print('~~~~ {} seconds ~~~~'.format(round(time.time() - start_time, 0)))
        print('Epoch: {}'.format(epoch))
        print(np.shape(train()))
