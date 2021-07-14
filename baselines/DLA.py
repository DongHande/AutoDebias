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
    if args.dataset == 'simulation':
        args.training_args = {'batch_size': 1024, 'epochs': 500, 'patience': 60, 'block_batch': [20, 500]}
        args.base_model_args = {'emb_dim': 10, 'learning_rate': 0.01, 'weight_decay': 5}
        args.position_model_args = {'learning_rate': 0.001, 'weight_decay': 0}
    else: 
        print('invalid arguments')
        os._exit()


def train_and_eval(train_data, val_data, test_data, device = 'cuda', model_class = MF, 
        base_model_args: dict = {'emb_dim': 64, 'learning_rate': 0.05, 'weight_decay': 0.05}, 
        position_model_args: dict = {'learning_rate': 0.05, 'weight_decay': 0.05}, 
        training_args: dict =  {'batch_size': 1024, 'epochs': 100, 'patience': 20, 'block_batch': [1000, 100]}): 
    train_position = train_data['position']
    train_rating = train_data['rating']

    # build data_loader. 
    train_loader = utils.data_loader.User(train_position, train_rating, u_batch_size=training_args['block_batch'][0], device=device)
    val_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(val_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)
    test_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(test_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)

    n_user, n_item = train_position.shape
    n_position = torch.max(train_position._values()).item() + 1

    base_model = MF(n_user, n_item, dim=base_model_args['emb_dim']).to(device)
    base_optimizer = torch.optim.SGD(base_model.parameters(), lr=base_model_args['learning_rate'], weight_decay=0)

    position_model = Position(n_position).to(device)
    position_optimizer = torch.optim.SGD(position_model.parameters(), lr=position_model_args['learning_rate'], weight_decay=0)

    # begin training
    stopping_args = Stop_args(patience=training_args['patience'], max_epochs=training_args['epochs'])
    early_stopping = EarlyStopping(base_model, **stopping_args)

    none_criterion = nn.MSELoss(reduction='none')

    for epo in range(early_stopping.max_epochs):
        for u_batch_idx, users in enumerate(train_loader.User_loader): 
            base_model.train()
            position_model.train()

            # observation data in this batch
            users_train, items_train, positions_train, y_train = train_loader.get_batch(users)
            first_index_row = torch.where(positions_train==0)[0] // n_position
            first_index_col = torch.where(positions_train==0)[0] % n_position
            
            y_hat = base_model(users_train, items_train)
            p_hat = position_model(positions_train)

            # users_train = users_train.view(-1, n_position)
            # items_train = items_train.view(-1, n_position)
            # positions_train = positions_train.view(-1, n_position)
            y_train = y_train.view(-1, n_position)

            y_hat = y_hat.view(-1, n_position)
            p_hat = p_hat.view(-1, n_position)

            y_hat_som = torch.softmax(y_hat, dim=-1) # softmax
            p_hat_som = torch.softmax(p_hat, dim=-1)

            IRW = torch.detach(y_hat_som[first_index_row, first_index_col].unsqueeze(dim=-1) * torch.reciprocal(y_hat_som))
            IPW = torch.detach(p_hat_som[first_index_row, first_index_col].unsqueeze(dim=-1) * torch.reciprocal(p_hat_som))

            cost_base = none_criterion(y_hat, y_train)
            # IPW = torch.ones(IPW.shape).to(device)
            loss_IPW = torch.sum(IPW * cost_base) + base_model_args['weight_decay'] * base_model.l2_norm(users_train, items_train)
            cost_position = none_criterion(p_hat, y_train)
            loss_IRW = torch.sum(IRW * cost_position) + position_model_args['weight_decay'] * position_model.l2_norm(positions_train)

            # y_train = y_train == 1
            # loss_IRW = - torch.sum(y_train * IRW * torch.log(p_hat_som)) + position_model_args['weight_decay'] * position_model.l2_norm(positions_train)
            # loss_IPW = - torch.sum(y_train * IPW * torch.log(y_hat_som)) + base_model_args['weight_decay'] * base_model.l2_norm(users_train, items_train)
            
            base_optimizer.zero_grad()
            loss_IPW.backward()
            base_optimizer.step()

            position_optimizer.zero_grad()
            loss_IRW.backward()
            position_optimizer.step()
        
        base_model.eval()
        with torch.no_grad():
            # training metrics
            train_pre_ratings = torch.empty(0).to(device)
            train_ratings = torch.empty(0).to(device)
            for u_batch_idx, users in enumerate(train_loader.User_loader): 
                users_train, items_train, positions_train, y_train = train_loader.get_batch(users)
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
    train, unif_train, validation, test = utils.load_dataset.load_dataset(data_name=args.dataset, type = 'list', seed = args.seed, device=device)
    train_and_eval(train, validation, test, device, base_model_args = args.base_model_args, position_model_args = args.position_model_args, training_args = args.training_args)
