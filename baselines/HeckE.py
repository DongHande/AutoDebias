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
        args.base_model_args = {'emb_dim': 10, 'dropout': 0.0, 'learning_rate': 0.01, 'imputaion_lambda': 0.001, 'weight_decay': 1}
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

    criterion = nn.MSELoss(reduction='sum')

    # train Heckman model
    heckman_model = MF(n_user, n_item, dim=base_model_args['emb_dim']).to(device)
    heckman_optimizer = torch.optim.SGD(heckman_model.parameters(), lr=base_model_args['learning_rate'], weight_decay=0)

    stopping_args = Stop_args(patience=training_args['patience'], max_epochs=training_args['epochs'])
    early_stopping = EarlyStopping(heckman_model, **stopping_args)

    for epo in range(early_stopping.max_epochs):
        training_loss = 0
        for u_batch_idx, users in enumerate(train_loader.User_loader): 
            heckman_model.train()

            # observation data in this batch
            users_train, items_train, positions_train, y_train = train_loader.get_batch(users)
            y_hat_obs = heckman_model(users_train, items_train)
            loss_obs = torch.sum((y_hat_obs - 1)**2)

            all_pair = torch.cartesian_prod(users, torch.arange(n_item).to(device))
            users_all, items_all = all_pair[:,0], all_pair[:,1]
            y_hat_all = heckman_model(users_all, items_all)
            loss_all = torch.sum((y_hat_all + 1)**2)

            loss = loss_obs + base_model_args['imputaion_lambda']  * loss_all # + 0.000 * heckman_model.l2_norm(users_all, items_all)

            heckman_optimizer.zero_grad()
            loss.backward()
            heckman_optimizer.step()

            training_loss += loss.item()
            
        # print('Epoch: {0:2d} / {1}, Traning: {2}'.
        #         format(epo, training_args['epochs'], training_loss))

        if early_stopping.check([-training_loss], epo):
            break

    print('Loading {}th epoch for heckman model'.format(early_stopping.best_epoch))
    heckman_model.load_state_dict(early_stopping.best_state)
    heckman_model.eval()

    # train click model
    click_model = MF_heckman(n_user, n_item, dim=base_model_args['emb_dim']).to(device)
    click_optimizer = torch.optim.SGD(click_model.parameters(), lr=base_model_args['learning_rate'], weight_decay=0)

    stopping_args = Stop_args(patience=training_args['patience'], max_epochs=training_args['epochs'])
    early_stopping = EarlyStopping(click_model, **stopping_args)

    for epo in range(early_stopping.max_epochs):
        training_loss = 0
        for u_batch_idx, users in enumerate(train_loader.User_loader): 
            click_model.train()

            # observation data in this batch
            users_train, items_train, positions_train, y_train = train_loader.get_batch(users)
            heckman_train = heckman_model(users_train, items_train)
            lam_train = (2 * math.pi)**(-0.5) * torch.exp(- heckman_train**2 / 2) / torch.sigmoid(1.7 * heckman_train)

            y_hat = click_model(users_train, items_train, lam_train)
            loss = criterion(y_hat, y_train) + base_model_args['weight_decay'] * click_model.l2_norm(users_train, items_train)

            click_optimizer.zero_grad()
            loss.backward()
            click_optimizer.step()

            training_loss += loss.item()
            
        click_model.eval()
        with torch.no_grad():
            # training metrics
            train_pre_ratings = torch.empty(0).to(device)
            train_ratings = torch.empty(0).to(device)
            for u_batch_idx, users in enumerate(train_loader.User_loader): 
                users_train, items_train, positions_train, y_train = train_loader.get_batch(users)
                heckman_train = heckman_model(users_train, items_train)
                lams = (2 * math.pi)**(-0.5) * torch.exp(- heckman_train**2 / 2) / torch.sigmoid(1.7 * heckman_train)

                pre_ratings = click_model(users_train, items_train, lams)
                train_pre_ratings = torch.cat((train_pre_ratings, pre_ratings))
                train_ratings = torch.cat((train_ratings, y_train))

            # validation metrics
            val_pre_ratings = torch.empty(0).to(device)
            val_ratings = torch.empty(0).to(device)
            for batch_idx, (users, items, ratings) in enumerate(val_loader):
                heckman_train = heckman_model(users, items)
                lams = (2 * math.pi)**(-0.5) * torch.exp(- heckman_train**2 / 2) / torch.sigmoid(1.7 * heckman_train)
                pre_ratings = click_model(users, items, lams)
                val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
                val_ratings = torch.cat((val_ratings, ratings))
            
        train_results = utils.metrics.evaluate(train_pre_ratings, train_ratings, ['MSE', 'NLL'])
        val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['MSE', 'NLL', 'AUC'])

        print('Epoch: {0:2d} / {1}, Traning: {2}, Validation: {3}'.
                format(epo, training_args['epochs'], ' '.join([key+':'+'%.3f'%train_results[key] for key in train_results]), 
                ' '.join([key+':'+'%.3f'%val_results[key] for key in val_results])))

        if early_stopping.check([val_results['AUC']], epo):
            break

    print('Loading {}th epoch for click model'.format(early_stopping.best_epoch))
    click_model.load_state_dict(early_stopping.best_state)

    # validation metrics
    val_pre_ratings_1 = torch.empty(0).to(device)
    val_ratings_1 = torch.empty(0).to(device)
    for batch_idx, (users, items, ratings) in enumerate(val_loader):
        heckman_train = heckman_model(users, items)
        lams = (2 * math.pi)**(-0.5) * torch.exp(- heckman_train**2 / 2) / torch.sigmoid(1.7 * heckman_train)
        pre_ratings = click_model(users, items, lams)
        val_pre_ratings_1 = torch.cat((val_pre_ratings_1, pre_ratings))
        val_ratings_1 = torch.cat((val_ratings_1, ratings))

    # test metrics
    test_users_1 = torch.empty(0, dtype=torch.int64).to(device)
    test_items_1 = torch.empty(0, dtype=torch.int64).to(device)
    test_pre_ratings_1 = torch.empty(0).to(device)
    test_ratings_1 = torch.empty(0).to(device)
    for batch_idx, (users, items, ratings) in enumerate(test_loader):
        heckman_train = heckman_model(users, items)
        lams = (2 * math.pi)**(-0.5) * torch.exp(- heckman_train**2 / 2) / torch.sigmoid(1.7 * heckman_train)
        pre_ratings = click_model(users, items, lams)
        test_users_1 = torch.cat((test_users_1, users))
        test_items_1 = torch.cat((test_items_1, items))
        test_pre_ratings_1 = torch.cat((test_pre_ratings_1, pre_ratings))
        test_ratings_1 = torch.cat((test_ratings_1, ratings))

    val_results_1 = utils.metrics.evaluate(val_pre_ratings_1, val_ratings_1, ['MSE', 'NLL', 'AUC'])
    test_results_1 = utils.metrics.evaluate(test_pre_ratings_1, test_ratings_1, ['MSE', 'NLL', 'AUC', 'Recall_Precision_NDCG@'], users=test_users_1, items=test_items_1)
    print('-'*30)
    print('The performance of validation set: {}'.format(' '.join([key+':'+'%.3f'%val_results_1[key] for key in val_results_1])))
    print('The performance of testing set: {}'.format(' '.join([key+':'+'%.3f'%test_results_1[key] for key in test_results_1])))
    print('-'*30)

    # train position model (DLA)
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
            loss_IPW = torch.sum(IPW * cost_base) + 5 * base_model_args['weight_decay'] * base_model.l2_norm(users_train, items_train)
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
    val_pre_ratings_2 = torch.empty(0).to(device)
    val_ratings_2 = torch.empty(0).to(device)
    for batch_idx, (users, items, ratings) in enumerate(val_loader):
        pre_ratings = base_model(users, items)
        val_pre_ratings_2 = torch.cat((val_pre_ratings_2, pre_ratings))
        val_ratings_2 = torch.cat((val_ratings_2, ratings))

    # test metrics
    test_users_2 = torch.empty(0, dtype=torch.int64).to(device)
    test_items_2 = torch.empty(0, dtype=torch.int64).to(device)
    test_pre_ratings_2 = torch.empty(0).to(device)
    test_ratings_2 = torch.empty(0).to(device)
    for batch_idx, (users, items, ratings) in enumerate(test_loader):
        pre_ratings = base_model(users, items)
        test_users_2 = torch.cat((test_users_2, users))
        test_items_2 = torch.cat((test_items_2, items))
        test_pre_ratings_2 = torch.cat((test_pre_ratings_2, pre_ratings))
        test_ratings_2 = torch.cat((test_ratings_2, ratings))

    val_results_2 = utils.metrics.evaluate(val_pre_ratings_2, val_ratings_2, ['MSE', 'NLL', 'AUC'])
    test_results_2 = utils.metrics.evaluate(test_pre_ratings_2, test_ratings_2, ['MSE', 'NLL', 'AUC', 'Recall_Precision_NDCG@'], users=test_users_2, items=test_items_2)
    print('-'*30)
    print('The performance of validation set: {}'.format(' '.join([key+':'+'%.3f'%val_results_2[key] for key in val_results_2])))
    print('The performance of testing set: {}'.format(' '.join([key+':'+'%.3f'%test_results_2[key] for key in test_results_2])))
    print('-'*30)

    alp = 0.4
    val_ratings = val_ratings_1
    val_pre_ratings = alp * val_pre_ratings_1 + (1-alp) * val_pre_ratings_2

    test_users = test_users_1
    test_items = test_items_1
    test_ratings = test_ratings_1
    test_pre_ratings = alp * test_pre_ratings_1 + (1-alp) * test_pre_ratings_2
    
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
