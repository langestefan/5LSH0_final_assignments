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


from re_id import calculate_distances


# network class
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # to store 2d feature vector results
        # store this in zip((x0,y0), label, prediction)
        self.feature_vectors_test = []
        self.feature_vectors_points_test = np.zeros((0, 2))  # to store 2D feature vectors for testdata
        self.feature_vectors_labels_test = np.zeros((0, 1), dtype=np.int8)  # to store groundtruth for testdata
        self.feature_vectors_pred_test = np.zeros((0, 1), dtype=np.int8)  # to store prediction for testdata

        self.feature_vectors_train = []
        self.feature_vectors_points_train = np.zeros((0, 2))  # to store 2D feature vectors for traindata
        self.feature_vectors_labels_train = np.zeros((0, 1), dtype=np.int8)  # to store groundtruth for traindata
        self.feature_vectors_pred_train = np.zeros((0, 1), dtype=np.int8)  # to store prediction for traindata

        self.testing_train_data = False

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
            self.feature_vectors_points_test = np.append(self.feature_vectors_points_test, x.numpy(), axis=0)
        else:
            self.feature_vectors_points_train = np.append(self.feature_vectors_points_train, x.detach().numpy(), axis=0)

        x = self.fc2(x)
        return x


# Plot and show
def plot(all_points):
    """
    Plot function from assignment document
    :param all_points: (digits, points, (x0, y0)) = (10, 20, 2)
    """
    colors = matplotlib.cm.Paired(np.linspace(0, 1, len(all_points)))
    fig, ax = plt.subplots(figsize=(7, 5))
    for (points, color, digit) in zip(all_points, colors, range(10)):
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

    # loop over all mini-batches
    for batch_id, (mini_batch, tlabel) in enumerate(train_data):
        # save training labels
        network.feature_vectors_labels_train = np.append(network.feature_vectors_labels_train,
                                                         np.transpose([tlabel.numpy()]), axis=0)

        # we input our data in N = <bs_train> samples at a time
        output = network(mini_batch)
        loss = lossf(output, tlabel)  # compute loss
        optimizer.zero_grad()  # make sure old gradients are set to 0 before the next step

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
            network.feature_vectors_labels_test = np.append(network.feature_vectors_labels_test,
                                                            np.transpose([labels.numpy()]), axis=0)

            output = network(data)
            loss += lossf(output, labels).item()

            # check if digit was correctly classified
            pred_label = torch.max(output, 1)[1].data.squeeze()

            # store predicted labels
            if not network.testing_train_data:
                network.feature_vectors_pred_test = np.append(network.feature_vectors_pred_test, pred_label)
            else:
                network.feature_vectors_pred_train = np.append(network.feature_vectors_pred_train, pred_label)

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
    bs_test = 1000  # 1000

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
        # to hold sorted 2d feature vectors for plotting test data
        sorted_features_plot = np.zeros((10, 20, 2))

        # to hold gallery images array(60000, 3) = (x0, y0, digit)
        gallery = np.zeros((0, 2))

        # we test once without training, then we complete the epoch loop
        test()  # test updated model

        # print 2D feature vector of last N=<bs_test> test samples
        # a = network.feature_vectors_ground_truth_test
        # concatenate feature vectors and labels into a single np array so it's easy to search through
        pts_labels_test = np.concatenate((network.feature_vectors_points_test, network.feature_vectors_labels_test),
                                         axis=1)

        # zip 2D feature vectors (x0, y0) with pred and groudtruth labels (test)
        network.feature_vectors_test = zip(network.feature_vectors_points_test, network.feature_vectors_labels_test,
                                           network.feature_vectors_pred_test)

        # sort data by label for plotting
        for id_x in range(10):
            sorted_features_plot[id_x] = pts_labels_test[np.where(pts_labels_test[:, 2] == id_x)][:20, :2]

        # plot 2D feature embedding space of test data
        # plot(sorted_features_plot)

        # 1 training epoch
        train()

        # get predictions for our gallery data
        # test_data_copy = test_data
        # test_data = train_data  # test data is now our entire training data set
        # network.testing_train_data = True
        # test()  # run a test with training data as test data
        # test_data = test_data_copy

        # zip 2D feature vectors (x0, y0) with groudtruth labels (train)
        network.feature_vectors_test = zip(network.feature_vectors_points_train, network.feature_vectors_labels_train,
                                           network.feature_vectors_pred_train)

        # calculate distances
        distances = calculate_distances(network.feature_vectors_points_train, network.feature_vectors_points_test[0])

        print(distances)

        # generate gallery = train set with predicted labels
        # for i, (x0, y0, predlabel) in enumerate(network.feature_vectors_prediction_test):
        #     print(i, x0, y0, predlabel)
