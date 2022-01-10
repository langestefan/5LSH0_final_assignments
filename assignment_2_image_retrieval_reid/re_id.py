import numpy as np
from scipy.spatial.distance import euclidean


def calculate_distances(data, point):
    """
    Calculate for given point the euclidean distance to the other N-1 samples
    :param data: Gallery data (x,y)
    :param point: Query point (x,y)
    :return: distance from point to the other N-1 points
    """
    size = np.size(data, 0)
    x0, y0 = point[0], point[1]
    distance = np.zeros(size)

    for idx, sample in enumerate(data):
        distance[idx] = euclidean([x0, y0], [data[idx, 0], data[idx, 1]])

    return distance


def query(queries, gallery, n_results):
    """

    :param queries: Array(N,2) of points we want to query
    :param gallery:
    :param n_results:
    :return:
    """