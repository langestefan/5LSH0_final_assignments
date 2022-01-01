import numpy as np


def relu(x):
    """
    Calculate element-wise Rectified Linear Unit (ReLU)
    :param x: Input array
    :return: Rectified output
    """
    return np.maximum(x, 0)


def sigmoid(x):
    """
    Calculate element-wise sigmoid
    :param x: Input array
    :return: Output 1/(1+exp(-x))
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    """
    Calculate element-wise sigmoid derivative
    :param x: Input array
    :return: Output sigm(x)*(1-sigm(x))
    """
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    """
    Compute softmax for each value in x
    :return: Vector, each value between 0-1
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def cross_entropy(predictions, labels):
    """

    :param predictions: Predicted values
    :param labels: Actual values
    :return: Cross-entropy (scalar between 0-1)
    """
    epsilon = 1e-12
