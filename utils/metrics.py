import numpy as np
from scipy.sparse import lil_matrix
import torch
import pandas as pd

import utils.data_loader
import cppimport.import_hook
import utils.ex as ex

# ex = cppimport.imp('ex')

def calc(n,m,ttuser,ttitem,pre,ttrating,atk=5):
    user=ttuser.cpu().detach().numpy()
    item=ttitem.cpu().detach().numpy()
    pre=pre.cpu().detach().numpy()
    rating=ttrating.cpu().numpy()
    posid=np.where(rating==1)
    posuser=user[posid]
    positem=item[posid]
    preall=np.ones((n,m))*(-1000000)
    preall[user,item]=pre
    id=np.argsort(preall,axis=1,kind='quicksort',order=None)
    id=id[:,::-1]
    id1=id[:,:atk]
    # print(id1)
    ans=ex.gaotest(posuser,positem,id1,id)
    # pre@k, re@k, NDCG, MRR, NDCG@k
    # print(ans)
    return [ans[0],ans[1],ans[4]]

def nll(vector_predict, vector_true):
    return -1 / vector_true.shape[0] * torch.sum(torch.log(1 + torch.exp(-vector_predict * vector_true))).item()

def auc(vector_predict, vector_true, device = 'cuda'): 
    pos_indexes = torch.where(vector_true == 1)[0].to(device)
    pos_whe=(vector_true == 1).to(device)
    sort_indexes = torch.argsort(vector_predict).to(device)
    rank=torch.zeros((len(vector_predict))).to(device)
    rank[sort_indexes] = torch.FloatTensor(list(range(len(vector_predict)))).to(device)
    rank = rank * pos_whe
    auc = (torch.sum(rank) - len(pos_indexes) * (len(pos_indexes) - 1) / 2)/ \
            (len(pos_indexes) * (len(vector_predict) - len(pos_indexes)))
    return auc.item()

def mse(vector_predict, vector_true): 
    mse = torch.mean((vector_predict - vector_true)**2)
    return mse.item()

def evaluate(vector_Predict, vector_Test, metric_names, users = None, items = None):
    global_metrics = {
        "AUC": auc,
        "NLL": nll,
        "MSE": mse, 
        'Recall_Precision_NDCG@': 5}

    results = {}
    for name in metric_names:
        if name != 'Recall_Precision_NDCG@':
            results[name] = global_metrics[name](vector_predict=vector_Predict,
                                                      vector_true=vector_Test)

    if 'Recall_Precision_NDCG@' in metric_names: 
        users_num = torch.max(users).item() + 1
        items_num = torch.max(items).item() + 1
        Recall_Precision_NDCG = calc(users_num, items_num, users, items, vector_Predict, vector_Test, atk=global_metrics['Recall_Precision_NDCG@'])
        results['Precision'] =  Recall_Precision_NDCG[0]
        results['Recall'] =  Recall_Precision_NDCG[1]
        results['NDCG'] =  Recall_Precision_NDCG[2]
        
    return results

