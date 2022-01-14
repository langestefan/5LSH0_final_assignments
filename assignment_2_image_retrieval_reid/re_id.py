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
    :param queries: Array(N,2) of points we want to query, UNKNOWN labels
    :param q_labels: Query labels
    :param gallery: All points, KNOWN labels
    :param g_labels: Gallery labels
    :param top_k: Closest top_k number of points we want to use for re-id
    :return: Array(n_queries, n_results). Labels for N = <n_results> from gallery ranked by distance to query points
    """
    ap_avg_sum = 0
    n_correct_re_id = 0
    n_queries = np.shape(queries)[0]
    g_labels_tile = np.tile(g_labels, (n_queries, 1))
    # compute distance between each gallery image and each query
    distance_matrix = cdist(gallery, queries)

    # get the indices of the topk lowest distances per query. argpartition is about half a minute faster
    # than argsort on my PC, but it does not sort the indices.
    # topk_ind = np.argpartition(distance_matrix, top_k, axis=0)[:top_k, :]
    topk_ind = distance_matrix.argsort(axis=0)[:top_k, :]
    # top20 labels from gallery, sorted by descending distance: array(20, 10000)
    topk_sorted_labels = np.take_along_axis(g_labels_tile, topk_ind, axis=0)

    # get the labels belonging to the topk gallery images
    for i_query in range(n_queries):
        ap_sum = 0  # AP sum for a query
        counted_tp = 0  # running number to count TP rate
        query_gt = q_labels[i_query]  # query ground truth (scalar)

        # accuracy
        topk_gallery_labels = g_labels[topk_ind[:, i_query]]
        n_tp = np.count_nonzero(query_gt == topk_gallery_labels)  # TP for this query
        n_correct_re_id += n_tp  # count total number of TP

        # implement mean Average Precision (mAP) metric. mAP = sum(APi) / n_queries for 0 < i < n_queries
        # see https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
        # can't compute mAP if there are no true positives! If n_tp = 0, then AP = 0 too.
        if n_tp != 0:
            for ii_topk in range(top_k):

                if q_labels[i_query] == topk_sorted_labels[ii_topk, i_query]:
                    counted_tp += 1
                    ap_sum += counted_tp / (ii_topk + 1)   # example: y, n, y, n = (1/1) + (0/2) + (2/3) + (0/4)

                # print('Matching: GT, labels, ap_sum', q_labels[i_query], topk_sorted_labels[:, i_query], ap_sum)

            ap = ap_sum / n_tp  # AP = ap_sum / TP   [0, 1]
            print('ap: ', ap)
            ap_avg_sum += ap

    # compute re-id and mean_ap result
    re_id_accuracy = n_correct_re_id / (top_k * n_queries)
    mean_ap = ap_avg_sum / n_queries

    return re_id_accuracy, topk_ind, mean_ap
