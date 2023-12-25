
import argparse
import os
import pandas as pd
import numpy as np



def load_text(data_path, text):
    files = [f for f in os.listdir(data_path) if f.endswith('.1D')]
    filenames = [name.split('.')[0] for name in files]
    file_idx = [name[:-9] for name in filenames]
    print('file name: ',file_idx)
    idx = pd.DataFrame({'FILE_ID': file_idx, 'file_name': files})
    logs = pd.merge(idx, text, how='left', on='FILE_ID')
    logs.to_csv('non_imaging_info_'+args.atlas+'.csv',index=False)
    return logs

def brain_graph(logs, atlas, path, data_folder):
    if not os.path.exists(path):
        os.makedirs(path)

    label = []
    for e in atlas['area'].values:
        if e.endswith('L'):
            label.append(0)
        elif e.endswith('R'):
            label.append(1)
        else:
            label.append(-1)
    atlas['label'] = label
    length = len(atlas)-1
    atlas.sort_values('index', inplace=True)
    atlas = atlas.reset_index().drop('level_0', axis=1)
    print(atlas)
    print('length:',length)

    adj = np.zeros([length, length])
    not_right = [i for i in range(length) if atlas['label'][i] != 1]
    not_left = [i for i in range(length) if atlas['label'][i] != 0]
    not_gb = [i for i in range(length) if atlas['label'][i] != -1]
    for idx in range(length):
        if atlas['label'][idx] == 0:
            adj[idx, not_left] = 1
        elif atlas['label'][idx] == 1:
            adj[idx, not_right] = 1
        elif atlas['label'][idx] == -1:
            adj[idx, not_gb] = 1
    node_ids = np.array_split(np.arange(1, length* 871 + 1), 871)
    adj_matrix = []

    for i in range(871):
        node_id = node_ids[i]
        for j in range(length):
            for k in range(length):
                if adj[j, k]:
                    adj_matrix.append([node_id[j], node_id[k]])
        pd.DataFrame(adj_matrix).to_csv(os.path.join(path1, 'ABIDE_'+args.atlas+'_A.txt'), index=False, header=False)
    print('Finish!')

    print('===================indicator===================')
    indicator = np.repeat(np.arange(1, 872), length)
    pd.DataFrame(indicator).to_csv(os.path.join(path1, 'ABIDE_'+args.atlas+'_graph_indicator.txt'), index=False, header=False)
    print('Indicator Finish!',indicator)

    graph_labels = logs[['label']]
    graph_labels.to_csv(os.path.join(path1, 'ABIDE_'+args.atlas+'_graph_labels.txt'), index=False, header=False)
    print('Finish!',graph_labels.shape)


    files = logs['file_name']
    node_att = pd.DataFrame([])
    for file in files:
        file_path = os.path.join(data_folder, file)
        rois = pd.read_csv(file_path, sep='\t').iloc[:110, :].T
        node_att = pd.concat([node_att, rois])
    node_att.to_csv(os.path.join(path1, 'ABIDE_'+args.atlas+'_node_attributes.txt'), index=False, header=False)
    cols = list(pd.read_csv(file_path, sep='\t').columns.values)
    for file in files:
        file_path = os.path.join(data_folder, file)
        temp_cols = list(pd.read_csv(file_path, sep='\t').columns.values)
        assert cols == temp_cols, 'Inconsistent order of brain regions in ABIDE pcp!'

    node_label = np.arange(length)
    node_labels = np.tile(node_label, 871)
    pd.DataFrame(node_labels).to_csv(os.path.join(path1, 'ABIDE_'+args.atlas+'_node_labels.txt'), index=False, header=False)
    print('Finish!',node_labels)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--atlas', type=str, default='ho', help='select: ho, ez, tt')
    args = parser.parse_args()

    atlas = pd.read_csv('./data/phenotypic/' + args.atlas + '_labels.csv', comment='#', header=None,names=['index', 'area'])
    root = './data/ABIDE_'+args.atlas+'/raw/'
    path = './temp/ABIDE_pcp/cpac/filt_global_'+args.atlas+'/'
    path1 = './data/ABIDE_' + args.atlas + '/raw/' #
    phenotypic = pd.read_csv('./temp/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')

    print("load {} logs".format(args.atlas))
    logs = load_text(path,phenotypic)
    logs['label'] = [2 - i for i in logs['DX_GROUP']]
    logs.to_csv(os.path.join( './data/phenotypic', 'log_'+args.atlas+'.csv'))


    brain_graph(logs, atlas, root, path)



