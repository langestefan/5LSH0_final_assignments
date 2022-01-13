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
import os
import cv2

from re_id import re_id_query


# network class
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # to store 2d feature vector results
        # store this in zip((x0,y0), label, prediction)
        # self.feature_vectors_test = []
        self.feature_vectors_test = np.zeros((0, 2))  # to store 2D feature vectors for testdata
        self.labels_test = np.zeros((0, 1), dtype=np.int8)  # to store groundtruth for testdata
        self.pred_test = np.zeros((0, 1), dtype=np.int8)  # to store prediction for testdata

        # self.feature_vectors_train = []
        self.feature_vectors_train = np.zeros((0, 2))  # to store 2D feature vectors for traindata
        self.labels_train = np.zeros((0, 1), dtype=np.int8)  # to store groundtruth for traindata
        self.pred_train = np.zeros((0, 1), dtype=np.int8)  # to store prediction for traindata

        self.testing_train_data = False
        self.complete_train_dataset = torch.empty((0, 1, 28, 28))

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
        if not self.training and not self.testing_train_data:
            self.feature_vectors_test = np.append(self.feature_vectors_test, x.detach().numpy(), axis=0)
        else:
            self.feature_vectors_train = np.append(self.feature_vectors_train, x.detach().numpy(), axis=0)

        x = self.fc2(x)
        return x


# Plot and show
def plot(mnist_points, hw_points):
    """
    Plot function from assignment document
    :param hw_points: Handwritten data feature vectors (digits, points, (x0, y0)) = (10, 1, 2)
    :param mnist_points: MNIST feature vectors (digits, points, (x0, y0)) = (10, 20, 2)
    """
    colors = matplotlib.cm.Paired(np.linspace(0, 1, len(mnist_points)))
    fig, ax = plt.subplots(figsize=(7, 5))

    for (points, color, digit) in zip(mnist_points, colors, range(10)):
        ax.scatter([item[0] for item in points],
                   [item[1] for item in points],
                   color=color, label='digit{}'.format(digit))

    for (point, color, digit) in zip(hw_points, colors, range(10)):
        ax.scatter([item[0] for item in point],
                   [item[1] for item in point],
                   color=color, marker='s')

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
        batch_size=batch_size_train_s, shuffle=False)
    # batch_size=batch_size_train_s, shuffle = true)

    test_d = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transform),
        batch_size=batch_size_test_s, shuffle=False)  # shuffle to false to track points over epochs

    return train_d, test_d


def concatenate_dataset():
    """
    To concatenate all training/testdata
    """
    # loop over all mini-batches
    for batch_id, (mini_batch, tlabel) in enumerate(train_data):
        network.complete_train_dataset = torch.cat((network.complete_train_dataset, mini_batch), dim=0)


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
        correct_n = 0
        total_n = 0

        # save training labels
        network.labels_train = np.append(network.labels_train,
                                         np.transpose([tlabel.numpy()]), axis=0)

        # we input our data in N = <bs_train> samples at a time
        output = network(mini_batch)
        train_loss = lossf(output, tlabel)  # compute train_loss
        optimizer.zero_grad()  # make sure old gradients are set to 0 before the next step

        # backpropagation
        train_loss.backward()
        optimizer.step()

        if batch_id % 100 == 0:
            train_loss = train_loss.item()
            current = batch_id * len(mini_batch)
            # print('train loss: {:f}   [{}/{}]'.format(np.round(train_loss, 5), current, size))

        # check if digit was correctly classified
        pred_label = torch.max(output, 1)[1].data.squeeze()

        # compute train batch accuracy
        for idd, label in enumerate(tlabel):
            if pred_label[idd] == label:
                correct_n += 1

        # batch accuracy/loss for report
        # batch_accuracy = correct_n/bs_train
        # train_loss = train_loss.item()
        # current = batch_id * len(mini_batch)
        # print('{:f}, {:f}, {:d}'.format(np.round(train_loss, 5), np.round(batch_accuracy, 5), int(current/64)))

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
    n_batches = len(test_data)
    test_loss = 0

    correct_n = np.zeros([10])  # we store correct_n for each digit individually
    total_n = np.zeros([10])

    # Make sure entire network is enabled and in test mode via .eval()
    network.eval()

    # with .no_grad() we make sure pytorch does not perform backprogation/gradient calculations
    with torch.no_grad():
        for data, labels in test_data:
            if not network.testing_train_data:
                network.labels_test = np.append(network.labels_test, np.transpose([labels.numpy()]), axis=0)
            else:
                network.labels_train = np.append(network.labels_train, np.transpose([labels.numpy()]), axis=0)

            # print('MNIST max: ', np.amax(data.numpy(), 0))

            output = network(data)
            test_loss += lossf(output, labels).item()

            # check if digit was correctly classified
            pred_label = torch.max(output, 1)[1].data.squeeze()

            # store predicted labels
            if not network.testing_train_data:
                network.pred_test = np.append(network.pred_test, pred_label)
            else:
                network.pred_train = np.append(network.pred_train, pred_label)

            for idd, label in enumerate(labels):
                total_n[label] += 1

                if pred_label[idd] == label:
                    correct_n[label] += 1

        # compute average loss by taking the sum over the loss per batch and divide by number of batches
        test_loss /= n_batches

        # compute correct/total for every digit
        correct_frac_digit = correct_n / total_n
        correct_frac_avg = np.sum(correct_n) / np.sum(total_n)

        print('Classification accuracy per label [0-9]: {}'.format(np.round(correct_frac_digit, 3)))
        print('Classification accuracy average   [0-9]: [{}]'.format(np.round(correct_frac_avg, 3)))
        print('Average test loss: [{}]'.format(np.round(test_loss, 5)))


def clear_test_variables():
    """
    clear persistent test variables
    """
    network.feature_vectors_test = np.zeros((0, 2))  # to store 2D feature vectors for testdata
    network.labels_test = np.zeros((0, 1), dtype=np.int8)  # to store groundtruth for testdata
    network.pred_test = np.zeros((0, 1), dtype=np.int8)  # to store prediction for testdata


def clear_train_variables():
    """
    clear persistent train variables
    """
    network.feature_vectors_train = np.zeros((0, 2))
    network.labels_train = np.zeros((0, 1), dtype=np.int8)
    network.pred_train = np.zeros((0, 1), dtype=np.int8)


def test_handwritten(images):
    """
    Run our handwritten digits throught the (trained) network
    :param images: Torch image stack torch.Size([10, 1, 28, 28]), handwritten input images to be tested/classified
    :return:
    """
    with torch.no_grad():
        labels_hw = np.linspace(0, 9, num=10, dtype=np.int8)
        # Make sure entire network is enabled and in test mode via .eval()
        network.eval()
        # check if digit was correctly classified
        output = network(images)
        pred_label = torch.max(output, 1)[1].data.squeeze().numpy()
        correct_frac_avg = np.sum(pred_label == labels_hw) / 10
        print('Classific HW: accuracy: [{0}], predicted label: {1}'.format(np.round(correct_frac_avg, 3), pred_label))

        print('--- start re-identification handwritten digits ---')
        re_id_accuracy, topk_indi, meanap = re_id_query(network.feature_vectors_test, labels_hw,
                                                        network.feature_vectors_train, network.labels_train, top_k=5)

        print('Re-id accuracy for handwritten digits: [{}]'.format(re_id_accuracy))
        print('mAP handwritten digits: [{}]'.format(np.round(meanap, 5)))

        # plot_dataset_images(topk_indi)


def plot_dataset_images(indices):
    """
    Plot function for question 9
    :param indices: Image indices to plot
    """
    plt.figure(figsize=(1, 1))

    for i in range(10):  # loop over digits 0-9
        plt.subplot(10, 6, (i * 6) + 1)  # 10 rows, 6 images per row
        plt.tight_layout()
        plt.imshow(images_hw[i, 0], cmap='gray', interpolation='none')
        # plt.title("GT label: {}".format(i))
        plt.xticks([])
        plt.yticks([])

        for ii in range(5):  # loop over top-k 0-5
            plt.subplot(10, 6, (i * 6) + ii + 2)  # 10 rows, 6 images per row
            plt.tight_layout()
            topk_ind = indices[ii, i]  # get each indice
            # plot gallery image belonging to that indice
            plt.imshow(network.complete_train_dataset[topk_ind, 0], cmap='gray', interpolation='none')
            # plt.title("Predicted: {}".format(network.pred_train[topk_ind]))  # include predicted label for gallery

            plt.xticks([])
            plt.yticks([])
    plt.show()


def import_hw_images(handwritten_image_dir):
    """
    Run our handwritten digits throught the (trained) network
    :param handwritten_image_dir: Directory where the handwritten images are stored
    :return:
    """
    image_list = os.listdir(handwritten_image_dir)
    image_stack = []
    img_norm = np.zeros((28, 28))

    for file_name in image_list:
        file_loc = os.path.join(handwritten_image_dir, file_name)
        img = cv2.imread(file_loc, cv2.IMREAD_GRAYSCALE)
        img_norm = cv2.normalize(img, img_norm, 0, 1, cv2.NORM_MINMAX)  # make sure values are between [0-1]
        image_stack.append(torch.tensor(img_norm).to(torch.float))

    images = torch.stack(image_stack, dim=0)
    mean = torch.mean(images, (0, 1, 2))
    std = torch.std(images, (0, 1, 2))
    images = (images[:, None, :, :] - mean) / std  # make sure all pixelvalues follow a standard normal distribution
    return images


if __name__ == '__main__':
    # network settings
    bs_test = 1000

    # hyperparameters
    LR = 0.01
    momentum = 0.9
    n_epochs = 10
    bs_train = 64

    # torch settings
    torch.backends.cudnn.enabled = False  # turn off GPU processing
    torch.manual_seed(1)  # to keep random numbers the same over executions

    # import data
    train_data, test_data = import_data(bs_train, bs_test)

    # create network object
    network = Network()

    # concatenate_dataset()  # for plotting

    # uncomment this line if we want to execute from pre-trained model
    # network.load_state_dict(torch.load('model.pth'))
    # print(network)

    # optimizer for backpropagation
    optimizer = optim.SGD(network.parameters(), lr=LR, momentum=momentum)

    # to keep track of runtime
    start_time = time.time()

    # nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.
    lossf = nn.CrossEntropyLoss()

    # import handwritten images
    images_hw = import_hw_images('processed')

    for epoch in range(n_epochs):
        print('--- Epoch: {0} - {1} seconds ---'.format(epoch, round(time.time() - start_time, 0)))
        mnist_fv_plot = np.zeros((10, 20, 2))  # to hold sorted 2d feature vectors for plotting test data
        hw_fv_plot = np.zeros((10, 1, 2))  # to hold sorted 2d feature vectors for plotting test data

        # we test once without training, then we complete the epoch loop
        clear_test_variables()  # clear 2d feature vectors
        test()  # test (updated) model

        # run a dry test with training data
        if epoch == 0:
            print('--- running dry test on training data ---')
            network.testing_train_data = True
            test_data_cpy = test_data
            test_data = train_data
            test()
            test_data = test_data_cpy
            network.testing_train_data = False

        # re-id testdata/gallery
        re_id_acc, topkind, mean_ap = re_id_query(network.feature_vectors_test, network.labels_test,
                                                  network.feature_vectors_train, network.labels_train, top_k=20)

        print('Re-id accuracy for test/train set MNIST: [{}]'.format(re_id_acc))
        print('mAP test/train set MNIST: [{}]'.format(np.round(mean_ap, 5)))

        # for plotting
        # points_test = np.concatenate((network.feature_vectors_test, network.labels_test),
        #                              axis=1)

        # re-id + test handwritten MNIST digits
        # clear_test_variables()
        # test_handwritten(images_hw)

        # for plotting handwritten feature vector points
        # hw_fv_plot = np.expand_dims(network.feature_vectors_test, axis=1)

        # sort data by label for plotting
        # for id_x in range(10):
        #     mnist_fv_plot[id_x] = points_test[np.where(points_test[:, 2] == id_x)][:20, :2]

        # plot 2D feature embedding space of test data
        # plot(mnist_fv_plot, hw_fv_plot)

        # 1 training epoch
        clear_train_variables()
        train()
