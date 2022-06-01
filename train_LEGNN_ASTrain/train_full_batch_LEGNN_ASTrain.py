import torch
import torch.nn as nn
import warnings
import os
import shutil
import copy
import logging
import time

import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
path_project = os.path.split(root_path)[0]
sys.path.append(root_path)
sys.path.append(path_project)

from utils.utils import set_random_seed, convert_to_gpu, get_n_params, get_optimizer, get_lr_scheduler, load_dataset, save_model_results, \
    generate_hetero_graph, add_connections_between_labels_nodes, get_train_predict_truth_idx, get_loss_func
from utils.full_batch_ASTrain_utils import train_model, evaluate_model, get_final_performance, get_pseudo_nodes_pseudo_labels
from model.LEGNN import LEGAT
from model.Classifier import Classifier
from utils.EarlyStopping import EarlyStopping

args = {
    'dataset': 'ogbn-arxiv',
    'predict_category': 'node',
    'seed': 0,
    'cuda': 0,
    'learning_rate': 0.002,
    'hidden_units': [180, 180, 180],
    'num_heads': 3,
    'input_drop': 0.1,
    'feat_drop': 0.0,
    'output_drop': 0.6,
    'use_attn_dst': False,
    'use_symmetric_norm': True,
    'residual': True,
    'norm': True,
    'train_select_rate': 0.5,
    'threshold': 0.7,
    'scale_factor': 10.0,
    'balance_factor': 0.5,
    'optimizer': 'adam',
    'weight_decay': 0,
    'epochs': 2000,
    'patience': 200,
    'num_runs': 10
}
args['device'] = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    train_accuracy_list, val_accuracy_list, test_accuracy_list = [], [], []

    for run in range(args['num_runs']):
        args['seed'] = run
        args['model_name'] = f'LEGNN_ASTrain_seed{args["seed"]}'

        set_random_seed(args['seed'])

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args['dataset']}/{args['model_name']}", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args['dataset']}/{args['model_name']}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        graph, labels, num_classes, train_idx, valid_idx, test_idx = load_dataset(root_path='../dataset', dataset_name=args['dataset'])

        graph = generate_hetero_graph(graph, num_classes)

        evaluate_graph = add_connections_between_labels_nodes(graph, labels, train_idx)

        legat = LEGAT(input_dim_dict={ntype: graph.nodes[ntype].data['feat'].shape[1] for ntype in graph.ntypes}, hidden_sizes=args['hidden_units'], etypes=graph.etypes,
                      ntypes=graph.ntypes, num_heads=args['num_heads'], residual=args['residual'], input_drop=args['input_drop'], feat_drop=args['feat_drop'],
                      output_drop=args['output_drop'], use_attn_dst=args['use_attn_dst'], norm=args['norm'], use_symmetric_norm=args['use_symmetric_norm'], full_batch=True)

        classifier = Classifier(n_hid=args['hidden_units'][-1] * args['num_heads'], n_out=num_classes)

        model = nn.Sequential(legat, classifier)
        logger.info(model)

        logger.info(f'the size of LEGNN_ASTrain parameters is {get_n_params(model)}.')

        logger.info(f'configuration is {args}')
        optimizer = get_optimizer(model, args['optimizer'], args['learning_rate'], args['weight_decay'])

        scheduler = get_lr_scheduler(optimizer, learning_rate=args['learning_rate'], t_max=args['epochs'])

        input_features = {ntype: graph.nodes[ntype].data['feat'] for ntype in graph.ntypes}

        graph, evaluate_graph, labels, model = convert_to_gpu(graph, evaluate_graph, labels, model, device=args['device'])
        for ntype in input_features:
            input_features[ntype] = convert_to_gpu(input_features[ntype], device=args['device'])

        graph, labels, model = convert_to_gpu(graph, labels, model, device=args['device'])

        save_model_folder = f"./save_model/{args['dataset']}/{args['model_name']}"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args['patience'], save_model_folder=save_model_folder,
                                       save_model_name=args['model_name'], logger=logger)

        loss_func = get_loss_func(dataset_name=args['dataset'])
        pseudo_loss_func = get_loss_func(dataset_name=args['dataset'], reduction='none')

        for epoch in range(args['epochs']):

            # compute the loss for train_predict_idx nodes, so get train predict and truth idx based on train_idx
            train_predict_idx, train_truth_idx = get_train_predict_truth_idx(train_idx, args['train_select_rate'])

            train_graph = add_connections_between_labels_nodes(graph, labels, train_truth_idx)

            train_loss, train_accuracy, logits = train_model(model, train_graph, copy.deepcopy(input_features), labels, train_predict_idx, args['predict_category'],
                                                             loss_func, dataset_name=args['dataset'])

            # compute the loss for pseudo nodes
            pseudo_label_loss, pseudo_node_idx, pseudo_node_labels = get_pseudo_nodes_pseudo_labels(logits=logits,
                                                                                                    evaluate_idx=torch.cat([valid_idx, test_idx], dim=0),
                                                                                                    loss_func=pseudo_loss_func,
                                                                                                    threshold=args['threshold'],
                                                                                                    epoch=epoch + 1,
                                                                                                    scale_factor=args['scale_factor'])

            loss = train_loss + args['balance_factor'] * pseudo_label_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step should be called after a batch has been used for training
            scheduler.step()

            val_loss, val_accuracy, test_loss, test_accuracy = evaluate_model(model, evaluate_graph, copy.deepcopy(input_features), labels, valid_idx, test_idx,
                                                                              args['predict_category'], loss_func, dataset_name=args['dataset'])

            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_loss:.4f}, accuracy {train_accuracy:.4f}, '
                f'valid loss: {val_loss:.4f}, accuracy {val_accuracy:.4f}, test loss: {test_loss:.4f}, accuracy {test_accuracy:.4f}')

            early_stop = early_stopping.step([('accuracy', val_accuracy, True)], model)
            if early_stop:
                break

        # load best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info('get final performance...')

        train_accuracy, val_accuracy, test_accuracy = get_final_performance(model, evaluate_graph, copy.deepcopy(input_features), labels, train_idx,
                                                                            valid_idx, test_idx, args['predict_category'], dataset_name=args['dataset'])

        logger.info(f'final train accuracy {train_accuracy:.4f}, valid accuracy {val_accuracy:.4f}, test accuracy {test_accuracy:.4f}')

        save_model_results(train_accuracy, val_accuracy, test_accuracy, save_result_folder=f"./results/{args['dataset']}", save_result_file_name=f"{args['model_name']}.json")

        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(val_accuracy)
        test_accuracy_list.append(test_accuracy)

        if run < args['num_runs'] - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

    logger.info(f'train accuracy: {train_accuracy_list}')
    logger.info(f'val accuracy: {val_accuracy_list}')
    logger.info(f'test accuracy: {test_accuracy_list}')
    logger.info(f'average train accuracy: {torch.mean(torch.Tensor(train_accuracy_list)):.4f} ± {torch.std(torch.Tensor(train_accuracy_list)):.4f}')
    logger.info(f'average val accuracy: {torch.mean(torch.Tensor(val_accuracy_list)):.4f} ± {torch.std(torch.Tensor(val_accuracy_list)):.4f}')
    logger.info(f'average test accuracy: {torch.mean(torch.Tensor(test_accuracy_list)):.4f} ± {torch.std(torch.Tensor(test_accuracy_list)):.4f}')
