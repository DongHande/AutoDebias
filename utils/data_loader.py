import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch

class Loader_list(Dataset): 
    def __init__(self, lst):
        self.lst = lst
    
    def __getitem__(self, index):
        return self.lst[index]

    def __len__(self):
        return len(self.lst)

class Block:
    def __init__(self, mat, u_batch_size = 1000, i_batch_size = 1000, device = 'cuda'): 
        self.mat = mat
        self.n_users, self.n_items = self.mat.shape
        self.User_loader = DataLoader(Loader_list(torch.arange(self.n_users).to(device)), batch_size=u_batch_size, shuffle=True, num_workers=0)
        self.Item_loader = DataLoader(Loader_list(torch.arange(self.n_items).to(device)), batch_size=i_batch_size, shuffle=True, num_workers=0)
    
    def get_batch(self, batch_user, batch_item, device = 'cuda'): 
        index_row = np.isin(self.mat._indices()[0].cpu().numpy(), batch_user.cpu().numpy())
        index_col = np.isin(self.mat._indices()[1].cpu().numpy(), batch_item.cpu().numpy())
        index = torch.tensor(np.where(index_row * index_col)[0]).to(device)
        # index_row = (mat._indices()[0][..., None] == batch_user[None, ...]).any(-1)
        # index_col = (mat._indices()[1][..., None] == batch_item[None, ...]).any(-1)
        # index = torch.where(index_row * index_col)[0]
        return self.mat._indices()[0][index], self.mat._indices()[1][index], self.mat._values()[index]
 
class User: 
    def __init__(self, mat_position, mat_rating, u_batch_size = 100, device = 'cuda'):
        self.mat_position = mat_position
        self.mat_rating = mat_rating
        self.n_users, self.n_items = self.mat_position.shape
        self.User_loader = DataLoader(Loader_list(torch.arange(self.n_users).to(device)), batch_size=u_batch_size, shuffle=True, num_workers=0)
    
    def get_batch(self, batch_user, device = 'cuda'): 
        index = np.isin(self.mat_position._indices()[0].cpu().numpy(), batch_user.cpu().numpy())
        index = torch.tensor(np.where(index)[0]).to(device)
        # index = (self.mat_pisition._indices()[0][..., None] == batch_user[None, ...]).any(-1)
        # index = torch.where(index_row)[0]
        return self.mat_position._indices()[0][index], self.mat_position._indices()[1][index], self.mat_position._values()[index], self.mat_rating._values()[index]

class Interactions(Dataset):
    """
    Hold data in the form of an interactions matrix.
    Typical use-case is like a ratings matrix:
    - Users are the rows
    - Items are the columns
    - Elements of the matrix are the ratings given by a user for an item.
    """

    def __init__(self, mat):
        self.mat = mat
        self.n_users, self.n_items = self.mat.shape

    def __getitem__(self, index):
        row = self.mat._indices()[0][index]
        col = self.mat._indices()[1][index]
        val = self.mat._values()[index]
        # return torch.tensor(int(row)), torch.tensor(int(col)), torch.tensor(val)
        return row, col, val

    def __len__(self):
        return self.mat._nnz()