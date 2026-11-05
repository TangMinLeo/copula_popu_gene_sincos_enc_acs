import json

import numpy as np
import pandas as pd
from copula_tools import *

# importance resampling using training data targent mariginal distributions and generated data
# data_gen takes the form of list of list, not a pandas dataframe
# data_ori and data_tar take the form of pandas dataframe
def importance_resamp(data_ori:pd.DataFrame, data_tar:pd.DataFrame, data_gen:list):
    
    # copular = CopulaScaler()
    # copular.fit(data_ori)
    # print(copular.ecdf_map)
    # ecdf_map = copular.ecdf_map

    training_marginals = {}
    for column in data_ori.columns:
        counts = data_ori[column].value_counts(normalize=True, sort=False)
        training_marginals[column] = counts.to_dict()
    # print(training_marginals)
    
    target_marginals = {}
    for column in data_tar.columns:
        counts = data_tar[column].value_counts(normalize=True, sort=False)
        target_marginals[column] = counts.to_dict()
    # print(target_marginals)
    
    data_gen_pd = pd.DataFrame(data_gen, columns=data_ori.columns)
    gene_marginals = {}
    for column in data_gen_pd.columns:
        counts = data_gen_pd[column].value_counts(normalize=True, sort=False)
        gene_marginals[column] = counts.to_dict()
    # print(gene_marginals)
        
        
    key_list = data_ori.columns
    weights = []
    count = 0
    for one_row in data_gen:
        prod = 1.0
        for i, one_value in enumerate(one_row):
            if one_value not in target_marginals[key_list[i]].keys():
                # print(one_value)
                prod = 0.0
                count += 1
                break
            else:
                prod *= target_marginals[key_list[i]][one_value]/gene_marginals[key_list[i]][one_value]
        weights.append(prod)
    print('len(weights): ', len(weights))
    
    print('max(weights): ', max(weights))
    print('min(weights): ', min(weights))
    print(count)
    
    normalized_weights = np.array(weights)/np.sum(weights)
    # 重采样
    resampled_idx = np.random.choice(len(data_gen), size=len(data_tar), p=normalized_weights)
    
    resampled_data = []
    for idx in resampled_idx:
        resampled_data.append(data_gen[idx])

    data_res = pd.DataFrame(resampled_data, columns=key_list)
    
    return data_res
    