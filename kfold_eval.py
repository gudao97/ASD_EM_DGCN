
import os
import torch
import pandas as pd
import numpy as np
from training import train_mlp, train_gcn, test_gcn
from models import  MultilayerPerceptron,DGCN
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch_geometric.data import DataLoader, Data
from torchinfo import summary



def kfold_mlp(data, args):

    args.times = 3
    args.least = 80
    args.patience = 60
    args.epochs = 200
    args.weight_decay = 0.1
    args.nhid = 256

    indices = np.arange(data.shape[0])
    kf = KFold(n_splits=10, random_state=args.seed, shuffle=True)
    val_kf = KFold(n_splits=10, shuffle=True)

    x = data[:, :-1]
    print("data：",x)
    print('X:  ',x.shape)
    y = data[:, -1]
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x, y) #
    args.num_features = x.shape[1]

    for repeat in range(args.times):
        print('%d times CV out of %d on training MLP...' % (repeat + 1, args.times))
        for i, (train_idx, test_idx) in enumerate(kf.split(indices)):
            print('》》》》》》》》》》》》MLP training %d fold...' % (i + 1))
            for count, (train_id, val_id) in enumerate(val_kf.split(train_idx)):
                if args.verbose:
                    print('%d val set out of 10' % (count + 1))
                val_idx = train_idx[val_id]
                train_idx1 = train_idx[train_id]
                fold_dir = os.path.join(args.check_dir, 'ABIDE_'+args.atlas+'_MLP', 'fold_%d' % (i + 1))
                if not os.path.exists(fold_dir):
                    os.makedirs(fold_dir)

                test_log = os.path.join(fold_dir, 'test_indices.txt')
                if not os.path.exists(test_log):
                    np.savetxt(test_log, test_idx, fmt='%d', delimiter=' ')
                else:
                    saved_indices = np.loadtxt(test_log, dtype=int)
                    assert np.array_equal(test_idx, saved_indices), \
                        'Something goes wrong with 10-fold cross-validation'

                model = MultilayerPerceptron(args).to(args.device)
                shape = (args.batch_size, args.num_features)
                summary(model, input_size=shape, depth=3, col_names=["input_size", "output_size", "num_params"], row_settings=["var_names"])
                print(model)

                opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                train_set = Subset(dataset, train_idx1) #
                val_set = Subset(dataset, val_idx)
                test_set = Subset(dataset, test_idx)
                assert len(set(list(train_idx1) + list(test_idx) + list(val_idx))) == x.shape[0], \
                    'Something goes wrong with 10-fold cross-validation'

                training_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
                validation_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
                test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

                best_model, best_val_acc, best_val_loss = train_mlp(model=model, train_loader=training_loader,
                                                                    val_loader=validation_loader, optimizer=opt,
                                                                    save_path=fold_dir, args=args)

                checkpoint = torch.load(os.path.join(fold_dir, '{}.pth'.format(best_model)))
                model.load_state_dict(checkpoint['net'])
                state = {'net': model.state_dict(), 'args': args}
                torch.save(state, os.path.join(fold_dir, 'num_{:d}_valloss_{:.6f}_pool_{:.3f}_epoch_{:d}_.pth'
                                               .format(count + 1, best_val_loss, args.pooling_ratio, best_model)))

def kfold_gcn(edge_index, edge_attr, num_samples, args):

    args.num_features = args.nhid // 2
    args.nhid = args.num_features // 2
    args.epochs = 3000
    args.patience = 1000
    args.weight_decay = 0.001
    args.least = 2000
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    indices = np.arange(num_samples)
    kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)



    result_df = pd.DataFrame([])
    test_result_acc = []
    test_result_loss = []
    test_result_spe = []
    test_result_sen = []
    test_result_pre = []
    test_result_auc = []

    for i, (train_idx, test_idx) in enumerate(kf.split(indices)):
        fold_path = os.path.join(args.data_dir, 'ABIDE_'+args.atlas+'_Further_Learned_Features', 'fold_%d' % (i + 1))
        work_path = os.path.join(args.check_dir, args.atlas+'_GCN')
        if not os.path.exists(work_path):
            os.makedirs(work_path)
        np.random.shuffle(train_idx)
        val_idx = train_idx[:len(train_idx)//10]
        train_idx = train_idx[len(train_idx)//10:]
        assert len(set(list(train_idx) + list(test_idx) + list(val_idx))) == num_samples, \
            'Something wrong in the CV'

        print('》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》Training '+args.atlas+'    '+ args.Method_GraphPool+' GCN on the [%d] fold《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《' % (i + 1))
        model = DGCN(args).to(args.device)
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
      
        feature_path = os.path.join(fold_path, 'features.txt')
        assert os.path.exists(feature_path), \
            'No further learned features found!'
        content = pd.read_csv(feature_path, header=None, sep='\t') 

        x = content.iloc[:, :-1].values 
        y = content.iloc[:, -1].values 

        x = torch.tensor(x, dtype=torch.float) 
        y = torch.tensor(y, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y) 

        train_mask = np.zeros(num_samples) 
        test_mask = np.zeros(num_samples)
        val_mask = np.zeros(num_samples)
        train_mask[train_idx] = 1
        test_mask[test_idx] = 1
        val_mask[val_idx] = 1

        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)

        assert np.array_equal(train_mask + val_mask + test_mask, np.ones_like(train_mask)), \
            'Something wrong with the cross-validation!'

        loader = DataLoader([data], batch_size=1)

        best_model = train_gcn(loader, model, optimizer, work_path, args)

        checkpoint = torch.load(os.path.join(work_path, '{}.pth'.format(best_model)))
        model.load_state_dict(checkpoint['net'])
        test_acc, test_loss,test_out,test_sen, test_spe,test_pre,test_auc = test_gcn(loader, model, args)

        result_df['fold_%d_' % (i + 1)] = test_out
        test_result_acc.append(test_acc)
        test_result_loss.append(test_loss)
        test_result_sen.append(test_sen)
        test_result_spe.append(test_spe)
        test_result_pre.append(test_pre)
        test_result_auc.append(test_auc)

        acc_val, loss_val, _, sen_val, spe_val ,pre_val,auc_val= test_gcn(loader, model, args, test=False)
        print('GCN {:0>2d} fold test set results, loss = {:.6f}, accuracy = {:.6f}'.format(i + 1, test_loss, test_acc))
        print('GCN {:0>2d} fold test set results, SEN = {:.6f}, SPE = {:.6f}'.format(i + 1, test_sen, test_spe))
        print('GCN {:0>2d} fold test set results, PRE = {:.6f}, AUC = {:.6f}'.format(i + 1, test_pre, test_auc))
        print('====================================================================================================')
        print('GCN {:0>2d} fold val set results, loss = {:.6f}, accuracy = {:.6f}'.format(i + 1, loss_val, acc_val))




        state = {'net': model.state_dict(), 'args': args}
        torch.save(state, os.path.join(work_path, 'fold_{:d}_test_{:.6f}_drop_{:.3f}_epoch_{:d}_.pth'
                                       .format(i + 1, test_acc, args.dropout_ratio, best_model)))
    result_path = args.result_dir
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    result_df.to_csv(os.path.join(result_path,
                                  'GCN_pool_%.3f_seed_%d_.csv' % (args.pooling_ratio, args.seed)),
                     index=False, header=True)

    print(args.atlas+'acc: [    %f    ]' % (sum(test_result_acc)/len(test_result_acc)))
    print(args.atlas+'sen: [    %f    ]' % (sum(test_result_sen)/len(test_result_sen)))
    print(args.atlas+'spe: [    %f    ]' % (sum(test_result_spe)/len(test_result_spe)))
    print(args.atlas+'auc: [    %f    ]' % (sum(test_result_auc)/len(test_result_auc)))


