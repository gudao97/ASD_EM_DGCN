
import os
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./data', help='save root')
parser.add_argument('--verbose', type=bool, default=True, )
parser.add_argument('--atlas',type=str,default='ho',help='select:ho, ez, tt')
args = parser.parse_args()



def load_text(data_path, text):

    files = [f for f in os.listdir(data_path) if f.endswith('.1D')]
    filenames = [name.split('.')[0] for name in files]
    file_idx = [name[:-8] for name in filenames]  # remove _rois_ho
    print('felie name:',file_idx)
    idx = pd.DataFrame({'FILE_ID': file_idx, 'file_name': files})
    logs = pd.merge(idx, text, how='left', on='FILE_ID')
    logs.to_csv('non_imaging_info_'+args.atlas+'.csv',index=False)
    return logs


if __name__ == '__main__':
    print('===================download ABIDEⅠ===================')
    path = os.path.join('./temp', 'ABIDE_pcp', 'cpac', 'filt_global_'+args.atlas)

    info_path = os.path.join(args.root, 'phenotypic')
    print(info_path)
    if not os.path.exists(info_path):
        os.makedirs(info_path)
    print('===================load {} label==================='.format(args.atlas))
    phenotypic = pd.read_csv(os.path.join('./temp', 'ABIDE_pcp', 'Phenotypic_V1_0b_preprocessed1.csv'))
    logs = load_text(path, phenotypic)
    logs['label'] = [2 - i for i in logs['DX_GROUP']]
    logs.to_csv(os.path.join(args.root, 'phenotypic', 'log_'+args.atlas+'.csv'))
