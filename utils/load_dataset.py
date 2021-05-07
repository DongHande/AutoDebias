import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

from torch.utils.data import Dataset, DataLoader
import torch

def seed_randomly_split(df, ratio, split_seed, shape):
    """
    Split based on a deterministic seed randomly
    """
    # Set the random seed for splitting
    np.random.seed(split_seed)

    # Randomly shuffle the data
    rows, cols, rating = df['uid'], df['iid'], df['rating']
    num_nonzeros = len(rows)
    permute_indices = np.random.permutation(num_nonzeros)
    rows, cols, rating = rows[permute_indices], cols[permute_indices], rating[permute_indices]

    # Convert to train/valid/test matrix
    idx = [int(ratio[0] * num_nonzeros), int(ratio[0] * num_nonzeros) + int(ratio[1] * num_nonzeros)]

    train = sp.csr_matrix((rating[:idx[0]], (rows[:idx[0]], cols[:idx[0]])),
                              shape=shape, dtype='float32')

    valid = sp.csr_matrix((rating[idx[0]:idx[1]], (rows[idx[0]:idx[1]], cols[idx[0]:idx[1]])),
                              shape=shape, dtype='float32')

    test = sp.csr_matrix((rating[idx[1]:], (rows[idx[1]:], cols[idx[1]:])),
                             shape=shape, dtype='float32')

    return train, valid, test

def load_dataset(data_name='yahooR3', type = 'explicit', unif_ratio = 0.05, seed=0, implicit=True, threshold=4, device = 'cuda'): 
    if type not in ['explicit', 'implicit', 'list']: 
        print('--------illegal type, please reset legal type.---------')
        return
    path = os.path.dirname(os.path.dirname(__file__)) + '/datasets/' + data_name
    if data_name == 'simulation': 
        user_df = pd.read_csv(path + '/user.txt', sep=',', header=None, names=['uid', 'iid', 'position', 'rating'])
    else: 
        user_df = pd.read_csv(path + '/user.txt', sep=',', header=None, names=['uid', 'iid', 'rating'])
    random_df = pd.read_csv(path + '/random.txt', sep=',', header=None, names=['uid', 'iid', 'rating'])

    if type == 'implicit': 
        user_df = user_df.drop(user_df[user_df['rating'] < threshold].index)

    # binirize the rating to -1/1
    user_df['rating'].loc[user_df['rating'] < threshold] = -1
    user_df['rating'].loc[user_df['rating'] >= threshold] = 1

    random_df['rating'].loc[random_df['rating'] < threshold] = -1
    random_df['rating'].loc[random_df['rating'] >= threshold] = 1
    
    m, n = max(user_df['uid']) + 1, max(user_df['iid']) + 1
    ratio = (unif_ratio, 0.05, 1-unif_ratio-0.05)
    unif_train, validation, test = seed_randomly_split(df=random_df, ratio=ratio, 
                                                       split_seed=seed, shape=(m, n))
    if type == 'list': 
        train_pos = sp.csr_matrix((user_df['position'], (user_df['uid'], user_df['iid'])), shape=(m, n), dtype='int64')
        train_rating = sp.csr_matrix((user_df['rating'], (user_df['uid'], user_df['iid'])), shape=(m, n), dtype='float32')
        train = {}
        train['position'] = sparse_mx_to_torch_sparse_tensor(train_pos).to(device)
        train['rating'] = sparse_mx_to_torch_sparse_tensor(train_rating).to(device)
    else: 
        train = sp.csr_matrix((user_df['rating'], (user_df['uid'], user_df['iid'])), shape=(m, n), dtype='float32')
        train = sparse_mx_to_torch_sparse_tensor(train).to(device)

    unif_train = sparse_mx_to_torch_sparse_tensor(unif_train).to(device)
    validation = sparse_mx_to_torch_sparse_tensor(validation).to(device)
    test = sparse_mx_to_torch_sparse_tensor(test).to(device)
    return train, unif_train, validation, test

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
