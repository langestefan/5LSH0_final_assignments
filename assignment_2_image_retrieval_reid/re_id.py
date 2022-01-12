import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist as cdist


def calculate_distances(data, point):
    """
    Calculate for given point the euclidean distance to the other N-1 samples
    :param data: Gallery data (x,y)
    :param point: Query point (x,y)
    :return: distance from point to the other N-1 points
    """
    size = np.size(data, 0)
    distance = np.zeros((size, 1))

    for idx, sample in enumerate(data):
        # distance[idx] = euclidean(point, sample)
        distance[idx] = np.linalg.norm(point - sample)

    return distance


def re_id_query(queries, q_labels, gallery, g_labels, top_k):
    """
    Do re-id query between all input queries and all input gallery samples
    :param g_labels: Query labels
    :param q_labels: Gallery labels
    :param top_k: Closest top_k number of points we want to use for re-id
    :param queries: Array(N,2) of points we want to query, UNKNOWN labels
    :param gallery: All points, KNOWN labels
    :return: Array(n_queries, n_results). Labels for N = <n_results> from gallery ranked by distance to query points
    """
    n_correct_re_id = 0
    n_queries = np.shape(queries)[0]

    # compute distance between each gallery image and each query
    print('--- compute re-identification distance matrix ---')
    distance_matrix = cdist(gallery, queries)

    # get the indices of the topk lowest distances per query
    topk_ind = np.argpartition(distance_matrix, top_k, axis=0)[:top_k, :]

    # get the labels belonging to the topk gallery images
    for index in range(np.shape(queries)[0]):
        topk_gallery_labels = g_labels[topk_ind[:, index]]
        n_correct_re_id += np.count_nonzero(q_labels[index] == topk_gallery_labels)

    re_id_accuracy = n_correct_re_id / (top_k * n_queries)

    return re_id_accuracy, topk_ind
