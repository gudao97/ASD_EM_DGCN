import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


def population_graph(args):

    ph = pd.read_csv(os.path.join(args.data_dir, 'phenotypic', 'log_' + args.atlas + '.csv'))

    ages = ph['AGE_AT_SCAN'].values
    ages = (ages - min(ages)) / (max(ages) - min(ages))


    sex_site = ['SEX', 'SITE_ID']
    text_info = ph[sex_site].values

    enc = OneHotEncoder()
    enc.fit(text_info)
    text_feature = enc.transform(text_info).toarray()
    sex_site_age = np.c_[text_feature, ages]

    adj = []
    att = []

    sim_matrix = cosine_similarity(sex_site_age)

    for i in range(871):
        for j in range(871):
            if sim_matrix[i, j] > 0.8 and i > j:
                adj.append([i, j])
                att.append(sim_matrix[i, j])

    adj = np.array(adj).T
    att = np.array([att]).T
    print('att',att.shape)

    if not os.path.exists(os.path.join(args.data_dir, 'population graph')):
        os.makedirs(os.path.join(args.data_dir, 'population graph'))
    pd.DataFrame(adj).to_csv(os.path.join(args.data_dir, 'population graph', 'ABIDE.adj'), index=False, header=False)
    pd.DataFrame(att).to_csv(os.path.join(args.data_dir, 'population graph', 'ABIDE.attr'), index=False, header=False)


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='root')
    parser.add_argument('--atlas', type=str, default='ho', help='select', choices=['ho', 'ez', 'tt'])
    args = parser.parse_args()
    population_graph(args)