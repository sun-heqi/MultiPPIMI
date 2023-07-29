
import logging
import random
from math import sqrt

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import AllChem
from scipy import stats

# from datasets import graph_data_obj_to_nx_simple, nx_to_graph_data_obj_simple

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)


def rmse(y, f):
    return sqrt(((y - f) ** 2).mean(axis=0))


def mse(y, f):
    return ((y - f) ** 2).mean(axis=0)


def pearson(y, f):
    return np.corrcoef(y, f)[0, 1]


def spearman(y, f):
    return stats.spearmanr(y, f)[0]


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    # ci = S / z
    return S / z


