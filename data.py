__author__ = "Stefan Weißenberger and Johannes Klicpera"
__license__ = "MIT"

import os

import numpy as np
from scipy.linalg import expm

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from io_data import load_dataset
from seeds import development_seed
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DATA_PATH = 'data'

def normalize_attributes(attr_matrix):
    epsilon = 1e-12
    if isinstance(attr_matrix, sp.csr_matrix):
        attr_norms = spla.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix.multiply(attr_invnorms[:, np.newaxis])
    else:
        attr_norms = np.linalg.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix * attr_invnorms[:, np.newaxis]
    return attr_mat_norm


def get_dataset(name: str, use_lcc: bool = True) -> InMemoryDataset:
    dataset = InMemoryDataset
    graph = load_dataset(name)
    graph.standardize(select_lcc=True)
    new_y = torch.LongTensor(graph.labels)
    data = Data(
            x=torch.FloatTensor(normalize_attributes(graph.attr_matrix).toarray()),
            edge_index=torch.LongTensor(graph.get_edgeid_to_idx_array().T),
            y=new_y,
            train_mask=torch.zeros(new_y.size(0), dtype=torch.bool),
            test_mask=torch.zeros(new_y.size(0), dtype=torch.bool),
            val_mask=torch.zeros(new_y.size(0), dtype=torch.bool)
        )
    dataset.data = data
    dataset.num_classes =len(np.unique(new_y))
    return dataset


def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
    rnd_state = np.random.RandomState(development_seed)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

    val_idx_tmp = [i for i in development_idx if i not in train_idx]

    val_idx = rnd_state.choice(val_idx_tmp, 500, replace=False)
    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data

