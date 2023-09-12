import gc

import numpy as np
import torch


def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
           N: the object number
           D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat



def graph_construct_kNN(X, k_neig=10, is_probH=False, m_prob=1.0):
    """
    param:
        X: N_object x feature_number
        k_neig: the number of neighbor expansion
    return:
        A: N_object x N_object
    """

    dis_mat = Eu_dis(X)
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))

    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            H[node_idx, center_idx] = 1.0

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0

    # print("KNN: H的top-k元素数量：", np.count_nonzero(H))
    return H



def graph_construct_xNN(X, delta=0.25):
    """
    param:
        X: N_object x feature_number
        k_neig: the number of neighbor expansion
    return:
        A: N_object x N_object
    """

    dis_mat = Eu_dis(X)     # 10000*10000
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    dis_mat = np.array(dis_mat/np.max(dis_mat))
    for center_idx in range(n_obj):
        H[center_idx, center_idx] = 1.0
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx].squeeze()     # 10000
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()   # 从小到大排列
        for idx in nearest_idx:
            if dis_vec[idx] <= delta:
                H[idx, center_idx] = 1.0
            else:
                break

    print("XNN: H的元素数量：", np.count_nonzero(H))
    return H


# 将超图关联矩阵H转为邻接矩阵A
def H2A(H):
    n_nodes = H.shape[0]
    ONES = torch.ones([n_nodes], device='cuda:0')

    A = torch.zeros([n_nodes, n_nodes], device='cuda:0')
    for i in range(n_nodes):
        edge_ind = torch.nonzero(H[i, :])  # N*1
        edge_ind = edge_ind.reshape(edge_ind.shape[0])  # N
        neighbor = H[:, edge_ind]  # N*e
        neighbor = torch.sum(neighbor, dim=1)
        A[i, :] = torch.where(neighbor > 0, ONES, A[i, :])
        A[:, i] = torch.where(neighbor.t() > 0, ONES.t(), A[:, i])

    gc.collect()
    torch.cuda.empty_cache()
    return A

