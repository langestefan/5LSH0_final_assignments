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


def re_id_query(queries, gallery, n_results=20):
    """
    Do re-id query between all input queries and all input gallery samples
    :param queries: Array(N,2) of points we want to query, UNKNOWN labels
    :param gallery: All points, KNOWN labels
    :param n_results: number N closest points to given query
    :return: Array(n_queries, n_results). Labels for N = <n_results> from gallery ranked by distance to query points
    """
    print('queries: {0}'.format(np.shape(queries)))
    print('gallery: {0}'.format(np.shape(gallery)))

    gallery_points, gallery_labels = zip(*gallery)
    n_queries = np.shape(queries)[0]
    results = []

    distances_cdist = cdist(gallery_points, queries)
    print('distances_cdist: {0}'.format(np.shape(distances_cdist)))
    print('distances_cdist: {0}'.format(distances_cdist))

    for idx, query in enumerate(queries):
        # calculate distance between query and all gallery points
        distances = calculate_distances(gallery_points, query)

        # concatenate distance and labels in an array so we can easily sort it
        combined = np.concatenate((gallery_points, gallery_labels, distances), axis=1)

        # order the array by distance (smallest --> largest) and keep only the smallest N=<n_results> rows
        d_sorted = combined[combined[:, 3].argsort()][:n_results, :]
        results.append(d_sorted)

        if idx % 10 == 0:
            print('Query: [{0}/{1}]'.format(idx, n_queries))

    print('results: {0}'.format(np.shape(results)))
    print('results: {0}'.format(results))
    return results
