import argparse

import numpy as np
import torch
import os
import pandas as pd
from population_graph import population_graph
from kfold_eval import kfold_mlp, kfold_gcn
from training import graph_pooling, extract
import logging
import datetime
import sys
import time

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=13, help='random seed,default=13')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--nhid', type=int, default=256, help='hidden size of MLP')
parser.add_argument('--atlas',type=str,default='ho',help='select atlas',choices=['ho','ez','tt','aal'])
parser.add_argument('--pooling_ratio', type=float, default=0.8, help='name')
parser.add_argument('--Method_GraphPool', type=str, default='HyperDrop', help='HyperDrop')
parser.add_argument('--edge-ratio', type=float, default=0.8, help='Pooling_ratio for edge_ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.01, help='dropout ratio')
parser.add_argument('--l2_regularization', type=float, default=0.001, help='l2_regularization')
parser.add_argument('--data_dir', type=str, default='./data', help='root')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--check_dir', type=str, default='./checkpoints', help='root of saved models')
parser.add_argument('--result_dir', type=str, default='./results', help='root of classification results')
parser.add_argument('--verbose', type=bool, default=True, help='print training details')
parser.add_argument("--model", type=str, default='HyperDrop', help='HyperDrop')
parser.add_argument('--num-convs', default=3, type=int)
args = parser.parse_args()
torch.manual_seed(args.seed)


class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    t = time.time()
    downsample_file_ho = os.path.join(args.data_dir, 'ABIDE_downsample','ABIDE_'+args.Method_GraphPool+'_ho_pool_{:.3f}_.txt'.format(args.pooling_ratio))
    print('training:',downsample_file_ho)
    if not os.path.exists(downsample_file_ho):
        print('Running graph pooling with pooling ratio = {:.3f}'.format(args.pooling_ratio))
        graph_pooling(args)
    downsample_ho = pd.read_csv(downsample_file_ho, header=None, sep='\t').values
    downsample_ho = downsample_ho[:,:-1]


    downsample_file_ez = os.path.join(args.data_dir, 'ABIDE_downsample','ABIDE_'+args.Method_GraphPool+'_ez_pool_{:.3f}_.txt'.format(args.pooling_ratio))
    if not os.path.exists(downsample_file_ez):
        print('Running graph pooling with pooling ratio = {:.3f}'.format(args.pooling_ratio))
        graph_pooling(args)
    downsample_ez = pd.read_csv(downsample_file_ez,header = None ,sep = '\t').values
    downsample_ez = downsample_ez[:, :-1]

    downsample_file_tt = os.path.join(args.data_dir, 'ABIDE_downsample','ABIDE_'+args.Method_GraphPool+'_tt_pool_{:.3f}_.txt'.format(args.pooling_ratio))
    if not os.path.exists(downsample_file_tt):
        print('Running graph pooling with pooling ratio = {:.3f}'.format(args.pooling_ratio))
        graph_pooling(args)
    downsample_tt = pd.read_csv(downsample_file_tt,header = None ,sep = '\t').values


    downsample = np.concatenate((downsample_ho,downsample_ez,downsample_tt),axis=1) # q_ho	k_ez	v_tt
    print(downsample.shape)

    # Single atlas
    # kfold_mlp(downsample_ho, args)
    # extract(downsample_ho, args)

    # Multi-atlas
    kfold_mlp(downsample, args)
    extract(downsample, args)


    adj_path = os.path.join(args.data_dir, 'population graph', 'ABIDE.adj')
    attr_path = os.path.join(args.data_dir, 'population graph', 'ABIDE.attr')
    if not os.path.exists(adj_path) or not os.path.exists(attr_path):
        print('you must construct population():')
        population_graph(args)

    edge_index = pd.read_csv(adj_path, header=None).values
    edge_attr = pd.read_csv(attr_path, header=None).values.reshape(-1)
    # kfold_gcn(edge_index, edge_attr, downsample_ho.shape[0], args) #single atlas
    kfold_gcn(edge_index, edge_attr, downsample.shape[0], args) # Multi-atlas
    print( 'Finally time: {:.2f}s'.format(time.time() - t))
    print('parameters=====  Pooling method{},Atlas:{},Pooling_ratio:{}'.format(args.Method_GraphPool,args.atlas,args.pooling_ratio))


    log_folder = './LOGS/ABIDE_'+args.atlas
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_filename = os.path.join(log_folder, f"logs_{current_time}.log")

