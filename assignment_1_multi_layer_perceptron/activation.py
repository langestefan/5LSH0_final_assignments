import numpy as np


def relu(x):
    """
    Calculate element-wise Rectified Linear Unit (ReLU)
    :param x: Input array
    :return: Rectified output
    """
    return np.maximum(x, 0)


def relu_deriv(x):
    """
    Calculate element-wise ReLU derivative
    See https://numpy.org/doc/stable/reference/generated/numpy.greater.html
    :param x: Input array
    :return: For each element in x: 1 if x>0, 0 if x<0, N length array
    """
    return np.greater(x, 0).astype(int)


def sigmoid(x):
    """
    Calculate element-wise sigmoid
    :param x: N length array
    :return: Output 1/(1+exp(-x)), N length array
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    """
    Calculate element-wise sigmoid derivative
    :param x: N length array
    :return: Output sigm(x)*(1-sigm(x)), N length array
    """
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    """
    Compute softmax for each value in x
    :param x: N length array
    :return: Softmax for ach value between 0-1, N length array
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def cross_entropy(predictions, labels):
    """
    Calculate the cross entropy.
    :param predictions: Predicted values, N length array
    :param labels: Actual values, N length array
    :return: Cross-entropy (scalar between 0-1)
    """
    n = np.size(labels)
    eps = 1e-12
    ce = -np.dot(labels, np.log(predictions + eps)) / n
    return ce
