import os
import numpy as np
import random

import torch
import torch.nn as nn

from model import *

import arguments

import utils.load_dataset
import utils.data_loader
import utils.metrics
from utils.early_stop import EarlyStopping, Stop_args

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def para(args): 
    if args.dataset == 'yahooR3': 
        args.training_args = {'batch_size': 1024, 'epochs': 500, 'patience': 60, 'block_batch': [6000, 500]}
        args.base_model_args = {'emb_dim': 10, 'learning_rate': 0.00001, 'weight_decay': 0}
        args.imputation_model_args = {'emb_dim': 10, 'learning_rate': 0.00001, 'weight_decay': 0}
    elif args.dataset == 'coat':
        args.training_args = {'batch_size': 128, 'epochs': 500, 'patience': 60, 'block_batch': [64, 64]}
        args.base_model_args = {'emb_dim': 10, 'learning_rate': 0.001, 'weight_decay': 10}
        args.imputation_model_args = {'emb_dim': 2, 'learning_rate': 0.001, 'weight_decay': 10}
    else: 
        print('invalid arguments')
        os._exit()
                

def train_and_eval(train_data, unif_train_data, val_data, test_data, device = 'cuda', 
        base_model_args: dict = {'emb_dim': 64, 'learning_rate': 0.01, 'weight_decay': 0.1}, 
        imputation_model_args: dict = {'emb_dim': 10, 'learning_rate': 0.1, 'weight_decay': 0.1}, 
        training_args: dict =  {'batch_size': 1024, 'epochs': 100, 'patience': 20, 'block_batch': [1000, 100]}): 
    # double robust model 

    # build data_loader. 
    train_loader = utils.data_loader.Block(train_data, u_batch_size=training_args['block_batch'][0], i_batch_size=training_args['block_batch'][1], device=device)
    val_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(val_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)
    test_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(test_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)

    # Naive Bayes propensity Estimator
    def Naive_Bayes_Propensity(train, unif): 
        # the implementation of naive bayes propensity
        P_Oeq1 = train._nnz() / (train.size()[0] * train.size()[1])

        y_unique = torch.unique(train._values())
        P_y_givenO = torch.zeros(y_unique.shape).to(device)
        P_y = torch.zeros(y_unique.shape).to(device)

        for i in range(len(y_unique)): 
            P_y_givenO[i] = torch.sum(train._values() == y_unique[i]) / torch.sum(torch.ones(train._values().shape).to(device))
            P_y[i] = torch.sum(unif._values() == y_unique[i]) / torch.sum(torch.ones(unif._values().shape).to(device))

        Propensity = P_y_givenO / P_y * P_Oeq1

        return y_unique, Propensity

    y_unique, Propensity = Naive_Bayes_Propensity(train_data, unif_train_data)
    InvP = torch.reciprocal(Propensity)
    
    # data shape
    n_user, n_item = train_data.shape

    # model and its optimizer. 
    base_model = MF(n_user, n_item, dim=base_model_args['emb_dim'], dropout=0).to(device)
    base_optimizer = torch.optim.SGD(base_model.parameters(), lr=base_model_args['learning_rate'], weight_decay=0)

    imputation_model = MF(n_user, n_item, dim=imputation_model_args['emb_dim'], dropout=0).to(device)
    imputation_optimizer = torch.optim.SGD(imputation_model.parameters(), lr=imputation_model_args['learning_rate'], weight_decay=0)
    
    # loss_criterion
    none_criterion = nn.MSELoss(reduction='none')
    sum_criterion = nn.MSELoss(reduction='sum')

    # begin training
    stopping_args = Stop_args(patience=training_args['patience'], max_epochs=training_args['epochs'])
    early_stopping = EarlyStopping(base_model, **stopping_args)
    for epo in range(early_stopping.max_epochs):
        for u_batch_idx, users in enumerate(train_loader.User_loader): 
            for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                # observation data in this batch
                users_train, items_train, y_train = train_loader.get_batch(users, items)
                weight = torch.ones(y_train.shape).to(device)
                for i in range(len(y_unique)): 
                    weight[y_train == y_unique[i]] = InvP[i]

                # step 1: update imptation error model
                imputation_model.train()

                e_hat = imputation_model(users_train, items_train) # imputation error
                e = y_train - base_model(users_train, items_train) # prediction error
                cost_e = none_criterion(e_hat, e) # the cost of error, i.e., the difference between imputaiton error and prediction error

                loss_imp = torch.sum(weight * cost_e) + imputation_model_args['weight_decay'] * imputation_model.l2_norm(users_train, items_train)
                
                imputation_optimizer.zero_grad()
                loss_imp.backward()
                imputation_optimizer.step()

                # step 2: update predition model
                base_model.train()

                # all pair data in this block
                all_pair = torch.cartesian_prod(users, items)
                users_all, items_all = all_pair[:,0], all_pair[:,1]

                y_hat_all = base_model(users_all, items_all)
                y_hat_all_detach = torch.detach(y_hat_all)
                g_all = imputation_model(users_all, items_all)

                loss_all = sum_criterion(y_hat_all, g_all + y_hat_all_detach) # sum(e_hat)

                # observation data
                y_hat_obs = base_model(users_train, items_train)
                y_hat_obs_detach = torch.detach(y_hat_obs)
                g_obs = imputation_model(users_train, items_train)

                e_obs = none_criterion(y_hat_obs, y_train)
                e_hat_obs = none_criterion(y_hat_obs, g_obs + y_hat_obs_detach)

                cost_obs = e_obs - e_hat_obs
                loss_obs = torch.sum(weight * cost_obs)

                loss_base = loss_all + loss_obs + base_model_args['weight_decay'] * base_model.l2_norm(users_all, items_all)

                base_optimizer.zero_grad()
                loss_base.backward()
                base_optimizer.step()
        
        base_model.eval()
        with torch.no_grad():
            # training metrics
            train_pre_ratings = torch.empty(0).to(device)
            train_ratings = torch.empty(0).to(device)
            for u_batch_idx, users in enumerate(train_loader.User_loader): 
                for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                    users_train, items_train, y_train = train_loader.get_batch(users, items)
                    pre_ratings = base_model(users_train, items_train)
                    train_pre_ratings = torch.cat((train_pre_ratings, pre_ratings))
                    train_ratings = torch.cat((train_ratings, y_train))

            # validation metrics
            val_pre_ratings = torch.empty(0).to(device)
            val_ratings = torch.empty(0).to(device)
            for batch_idx, (users, items, ratings) in enumerate(val_loader):
                pre_ratings = base_model(users, items)
                val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
                val_ratings = torch.cat((val_ratings, ratings))
            
        train_results = utils.metrics.evaluate(train_pre_ratings, train_ratings, ['MSE', 'NLL'])
        val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'NLL', 'AUC'])

        print('Epoch: {0:2d} / {1}, Traning: {2}, Validation: {3}'.
                format(epo, training_args['epochs'], ' '.join([key+':'+'%.3f'%train_results[key] for key in train_results]), 
                ' '.join([key+':'+'%.3f'%val_results[key] for key in val_results])))

        if early_stopping.check([val_results['AUC']], epo):
            break

    # testing loss
    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    base_model.load_state_dict(early_stopping.best_state)

    # validation metrics
    val_pre_ratings = torch.empty(0).to(device)
    val_ratings = torch.empty(0).to(device)
    for batch_idx, (users, items, ratings) in enumerate(val_loader):
        pre_ratings = base_model(users, items)
        val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
        val_ratings = torch.cat((val_ratings, ratings))

    # test metrics
    test_users = torch.empty(0, dtype=torch.int64).to(device)
    test_items = torch.empty(0, dtype=torch.int64).to(device)
    test_pre_ratings = torch.empty(0).to(device)
    test_ratings = torch.empty(0).to(device)
    for batch_idx, (users, items, ratings) in enumerate(test_loader):
        pre_ratings = base_model(users, items)
        test_users = torch.cat((test_users, users))
        test_items = torch.cat((test_items, items))
        test_pre_ratings = torch.cat((test_pre_ratings, pre_ratings))
        test_ratings = torch.cat((test_ratings, ratings))

    val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'NLL', 'AUC'])
    test_results = utils.metrics.evaluate(test_pre_ratings, test_ratings, ['MSE', 'NLL', 'AUC', 'Recall_Precision_NDCG@'], users=test_users, items=test_items)
    print('-'*30)
    print('The performance of validation set: {}'.format(' '.join([key+':'+'%.3f'%val_results[key] for key in val_results])))
    print('The performance of testing set: {}'.format(' '.join([key+':'+'%.3f'%test_results[key] for key in test_results])))
    print('-'*30)
    return val_results,test_results

if __name__ == "__main__": 
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train, unif_train, validation, test = utils.load_dataset.load_dataset(data_name=args.dataset, type = 'explicit', seed = args.seed, device=device)
    train_and_eval(train, unif_train, validation, test, device, base_model_args = args.base_model_args, imputation_model_args = args.imputation_model_args, training_args = args.training_args)
