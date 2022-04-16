import numpy as np


def relu(x):
    """
    Calculate element-wise Rectified Linear Unit (ReLU)
    :param x: Input array
    :return: Rectified output
    """
    return np.maximum(x, 0)


def relu_bw(x):
    """
    Calculate element-wise ReLU derivative
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


def sigmoid_bw(x):
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
    max_val = 700
    # this custom implementation gives overflow errors when used with momentum, so we clip...
    np.clip(x, -max_val, max_val, out=x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def softmax_bw(softmax_output, label):
    """
    Cross entropy error with softmax output, backpropagation of deritivative w.r.t the inputs to the softmax function.
    dE/dzk = softmax_output_k - label_k
    :param softmax_output: Softmax output vector of length 10.
    :param label: One-hot encoded label vector of length 10. Example: [0 0 0 0 0 0 1 0 0 0] = 7
    :return: softmax_output - label
    """
    return softmax_output - label


def cross_entropy(predictions, label):
    """
    Calculate the cross entropy.
    :param predictions: Predicted values, N length array
    :param label: One-hot encoded label vector. Example: [0 0 0 0 0 0 1 0 0 0] = 7
    :return: Cross-entropy
    """
    n = np.size(label)
    eps = 1e-12 # for numerical stability
    ce = -np.dot(label, np.log(predictions + eps)) / n
    return ce

