
import os
import torch
import time
import pandas as pd
import numpy as np
import shutil
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from models import GPModel, MultilayerPerceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import torch.nn.functional as F

import torch.nn as nn

def graph_pooling(args):
    torch.manual_seed(args.seed)

    abide_dataset = TUDataset(args.data_dir, name='ABIDE_'+args.atlas, use_node_attr=True)
    args.num_classes = abide_dataset.num_classes
    args.num_features = abide_dataset.num_features
    args.num_node_features = abide_dataset.num_node_features
    args.num_edge_features = abide_dataset.num_edge_features
    gp = GPModel(args).to(args.device)
    abide_loader = DataLoader(abide_dataset, batch_size=args.batch_size, shuffle=False)
    downsample = []
    label = []
    for i, data in enumerate(abide_loader):
        data = data.to(args.device)
        downsample += gp(data).cpu().detach().numpy().tolist()
        label += data.y.cpu().detach().numpy().tolist()
    downsample_df = pd.DataFrame(downsample)
    downsample_df['label'] = label
    downsample_dir = os.path.join(args.data_dir, 'ABIDE_downsample')
    if not os.path.exists(downsample_dir):
        os.makedirs(downsample_dir)
    downsample_file = os.path.join(downsample_dir,  'ABIDE_'+args.Method_GraphPool+'_'+args.atlas+'_pool_%.3f_.txt' % args.pooling_ratio) #
    downsample_df.to_csv(downsample_file, index=False, header=False, sep='\t') #

    del gp
    del data
    del abide_dataset
    del abide_loader
    del downsample_df
    torch.cuda.empty_cache()


def test_mlp(model, loader, args):

    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data_x, data_y in loader:
        data_x, data_y = data_x.to(args.device), data_y.to(args.device)
        out, _ = model(data_x)
        pred = (out > 0).long()
        correct += pred.eq(data_y).sum().item()
        loss_func = nn.BCEWithLogitsLoss()
        loss_test += loss_func(out, data_y.float()).item()

    return correct / len(loader.dataset), loss_test


def train_mlp(model, train_loader, val_loader, optimizer, save_path, args):

    min_loss = 1e10
    max_acc = 0
    patience_cnt = 0
    val_loss_values = []
    val_acc_values = []
    best_epoch = 0
    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        for i, (data_x,data_y) in enumerate(train_loader):
            optimizer.zero_grad()
            data_x, data_y = data_x.to(args.device), data_y.to(args.device)
            out, _ = model(data_x)
            loss_func = nn.BCEWithLogitsLoss()
            loss = loss_func(out, data_y.float())
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pred = (out > 0).long()
            correct += pred.eq(data_y).sum().item()
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = test_mlp(model, val_loader, args)
        if args.verbose:
            print('\r', 'Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
                  'acc_train: {:.6f}'.format(acc_train), 'time: {:.2f}s'.format(time.time() - t), end='', flush=True)

        val_loss_values.append(loss_val)
        val_acc_values.append(acc_val)
        if epoch < args.least:
            continue
        if val_loss_values[-1] <= min_loss:
            model_state = {'net': model.state_dict(), 'args': args}
            torch.save(model_state, os.path.join(save_path, '{}.pth'.format(epoch)))
            min_loss = val_loss_values[-1]
            max_acc = val_acc_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break


        files = [f for f in os.listdir(save_path) if f.endswith('.pth')]
        for f in files:
            if f.startswith('num'):
                continue
            epoch_nb = int(f.split('.')[0])
            if epoch_nb != best_epoch:
                os.remove(os.path.join(save_path, f))
    if args.verbose:
        print('\n finish: {:.2f}s'.format(time.time() - t))

    return best_epoch, max_acc, min_loss


def extract(data, args, least_epochs=100):
    x = data[:, :-1]
    y = data[:, -1]
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x, y)
    for i in range(10):
        fold_dir = os.path.join(args.check_dir, 'ABIDE_'+args.atlas+'_MLP', 'fold_%d' % (i + 1))
        files = os.listdir(fold_dir)
        max_epoch = 0
        best_model = None

        for f in files:
            if f.endswith('.pth') and f.startswith('num_'):
                acc = float(f.split('_')[3])
                epoch_num = int(f.split('_')[-2])
                if epoch_num > max_epoch:
                    max_epoch = epoch_num
                    best_model = f

        assert best_model is not None, \
            'Cannot find the trained model. Maybe the least_epochs is too large.'

        if args.verbose:
            print('从模型中提取信息 {}'.format(fold_dir + '/' + best_model))
        checkpoint = torch.load(os.path.join(fold_dir, best_model))
        model_args = checkpoint['args']
        dataloader = DataLoader(dataset, batch_size=model_args.batch_size, shuffle=False)
        model = MultilayerPerceptron(model_args).to(model_args.device)
        model.load_state_dict(checkpoint['net'])
        model.eval()
        feature_matrix = []
        label = []

        correct = 0
        for data_x, data_y in dataloader:
            data_x, data_y = data_x.to(args.device), data_y.to(args.device)
            out, features = model(data_x)
            feature_matrix += features.cpu().detach().numpy().tolist()
            pred = (out > 0).long()
            correct += pred.eq(data_y).sum().item()
            label += data_y.cpu().detach().numpy().tolist()

        fold_feature_matrix = np.array(feature_matrix)

        features = pd.DataFrame(fold_feature_matrix)
        features['label'] = label
        feature_path = os.path.join(args.data_dir,'ABIDE_'+args.atlas+'_Further_Learned_Features', 'fold_%d' % (i + 1))
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        features.to_csv(os.path.join(feature_path, 'features.txt'), header=False, index=False, sep='\t')
        shutil.copyfile(os.path.join(fold_dir, 'test_indices.txt'),
                        os.path.join(feature_path, 'test_indices.txt'))

    print('finish')
    print('feautres was saved: features.txt')

def test_gcn(dataloader, model, args, test=True):
    model.eval()
    loss_test = 0.0
    correct = 0.0
    total = 0
    output = []
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    auc = 0

    for data in dataloader:
        data = data.to(args.device)
        out, _,l2 = model(data.x, data.edge_index, data.edge_attr)
        output += out.cpu().detach().numpy().tolist()
        criterion = nn.BCEWithLogitsLoss()
        if test:

            loss_test += criterion(out[data.test_mask if test else data.val_mask],
                                   data.y[data.test_mask if test else data.val_mask].float()).item()+l2

            pred = (out[data.test_mask if test else data.val_mask] > 0).long()
            correct += pred.eq(data.y[data.test_mask if test else data.val_mask]).sum().item()
            total += data.test_mask.sum().item() if test else data.val_mask.sum().item()

            y_true = data.y[data.test_mask if test else data.val_mask].cpu().detach().numpy()
            y_pred = out[data.test_mask if test else data.val_mask].cpu().detach().numpy()
            auc = roc_auc_score(y_true, y_pred)

            true_positive += ((pred == 1) & (data.y[data.test_mask if test else data.val_mask] == 1)).sum().item()
            true_negative += ((pred == 0) & (data.y[data.test_mask if test else data.val_mask] == 0)).sum().item()
            false_positive += ((pred == 1) & (data.y[data.test_mask if test else data.val_mask] == 0)).sum().item()
            false_negative += ((pred == 0) & (data.y[data.test_mask if test else data.val_mask] == 1)).sum().item()

        else:
            loss_test += criterion(out[data.test_mask if test else data.val_mask],
                                   data.y[data.test_mask if test else data.val_mask].float()).item()+l2# 不加正则化需要删除l2
            pred = (out[data.test_mask if test else data.val_mask] > 0).long()
            correct += pred.eq(data.y[data.test_mask if test else data.val_mask]).sum().item()
            total += data.test_mask.sum().item() if test else data.val_mask.sum().item()

            y_true = data.y[data.test_mask if test else data.val_mask].cpu().detach().numpy()
            y_pred = out[data.test_mask if test else data.val_mask].cpu().detach().numpy()
            auc = roc_auc_score(y_true, y_pred)

            true_positive += ((pred == 1) & (data.y[data.test_mask if test else data.val_mask] == 1)).sum().item()
            true_negative += ((pred == 0) & (data.y[data.test_mask if test else data.val_mask] == 0)).sum().item()
            false_positive += ((pred == 1) & (data.y[data.test_mask if test else data.val_mask] == 0)).sum().item()
            false_negative += ((pred == 0) & (data.y[data.test_mask if test else data.val_mask] == 1)).sum().item()

    accuracy = correct / total
    loss = loss_test / len(dataloader) +l2
    if true_positive + false_negative == 0:
        sensitivity = 0.0
    else:
        sensitivity = true_positive / (true_positive + false_negative) # SEN
    if true_negative + false_positive == 0:
        specificity = 0.0
    else:
        specificity = true_negative / (true_negative + false_positive) # SPE
    if true_positive + false_positive == 0:
        Pre = 0.0
    else:
        Pre = true_positive / (true_positive + false_positive)
    return accuracy,loss,output, sensitivity, specificity, Pre ,auc


os.environ['CUDNN_MODE'] = 'RNN_MODE_PERSISTENT'
torch.backends.cudnn.enabled = False

def train_gcn(dataloader, model, optimizer, save_path, args):

    min_loss = 1e10
    patience_cnt = 0
    loss_set = []
    acc_set = []

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []


    best_epoch = 0
    num_epoch = 0
    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        num_epoch += 1
        for i, data in enumerate(dataloader): #
            optimizer.zero_grad()
            data = data.to(args.device)
            data.x = data.x.to(args.device).requires_grad_()
            out, features ,l2= model(data.x, data.edge_index, data.edge_attr)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(out[data.train_mask], data.y[data.train_mask].float())+l2
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            train_losses.append(loss_train)
            pred = (out[data.train_mask] > 0).long()
            correct += pred.eq(data.y[data.train_mask]).sum().item()

        acc_train = correct / data.train_mask.sum().item()
        train_accuracies.append(acc_train)
        acc_val, loss_val, _  , sen_val , spe_val ,pre_val,auc_val= test_gcn(dataloader, model, args, test=False)
        val_losses.append(loss_val)
        val_accuracies.append(acc_val)
        if args.verbose:
            print('\r', 'Epoch: {:06d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
                  'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
                  'acc_val: {:.6f}'.format(acc_val),'auc_val: {:.6f}'.format(auc_val),  'time: {:.2f}s'.format(time.time() - t), flush=True, end='')

        loss_set.append(loss_val)
        acc_set.append(acc_val)
        if epoch < args.least:
            continue
        if loss_set[-1] < min_loss:
            model_state = {'net': model.state_dict(), 'args': args}
            torch.save(model_state, os.path.join(save_path, '{}.pth'.format(epoch)))
            min_loss = loss_set[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt == args.patience:
            break

        files = [f for f in os.listdir(save_path) if f.endswith('.pth')] # 对每个模型文件执行以下操作
        for f in files:
            if f.startswith('fold'):
                continue
            epoch_nb = int(f.split('.')[0])
            if epoch_nb != best_epoch:
                try:
                    os.remove(os.path.join(save_path, f))
                except PermissionError:
                    print(f"Failed to delete file: {f}. It may be in use by another process.")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(args.atlas+args.Method_GraphPool+' Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    if args.verbose:
        print('\n   finish: {:.2f}'.format(time.time() - t))

    return best_epoch
