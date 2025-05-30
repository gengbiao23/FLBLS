import numpy as np
from constructW import constructW

def construct_L(X, gnd):
    """
    Constructs the graph Laplacian matrix L.
    X: Feature matrix (n_samples x n_features)
    gnd: Ground truth labels (n_samples,)
    """
    # 构建邻接矩阵Q
    options = {}
    options['NeighborMode'] = 'Supervised'
    options['gnd'] = gnd
    options['WeightMode'] = 'HeatKernel'
    options['t'] = 1.0  # 热核参数
    Q = constructW(X, options)

    # 计算度矩阵D
    aa = np.sum(Q, axis=0)
    DDD = np.diag(aa)

    # 拉普拉斯矩阵 L = D - Q
    L = DDD - Q
    return L
