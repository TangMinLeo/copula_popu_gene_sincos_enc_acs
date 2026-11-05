import json

import numpy as np
import pandas as pd
from copula_tools import *

f = open('./training/DDPM_cond_gene_nhts_3/training_sample_50000.json', 'r', encoding='utf')
training_data = json.load(f)
f.close()

data_train = pd.DataFrame(training_data, columns=['年龄（分段）', '性别', '家庭收入', '房屋类型', '家庭是否有车', '是否有驾照',
                                                  '每周工作天数', '工作类型', '学生类别', '家庭人数', '是否有小孩', '通勤方式',
                                                  '通勤出发时间'])
copular = CopulaScaler()
copular.fit(data_train)
# print(copular.ecdf_map)
ecdf_map = copular.ecdf_map

training_marginals = {}
for key in ecdf_map.keys():
    one_dim = ecdf_map[key]
    one_dim_dict = {}
    for i in range(len(one_dim)):
        if i == 0:
            one_dim_dict[one_dim[i][0]] = one_dim[i][1]
        else:
            one_dim_dict[one_dim[i][0]] = one_dim[i][1] - one_dim[i-1][1]

    training_marginals[key] = one_dim_dict
print(training_marginals)


ff = open('./copula_popu_gene_nths_3_1000_1.json', 'r', encoding='utf')
try_gene = json.load(ff)
ff.close()

ff = open('./copula_popu_gene_nths_3_10000.json', 'r', encoding='utf')
gene_ddpm_2 = json.load(ff)
ff.close()

data_ddpm_2 = pd.DataFrame(gene_ddpm_2, columns=['年龄（分段）', '性别', '家庭收入', '房屋类型', '家庭是否有车', '是否有驾照',
                                                  '每周工作天数', '工作类型', '学生类别', '家庭人数', '是否有小孩', '通勤方式',
                                                  '通勤出发时间'])

target_marginals = {}
for column in data_ddpm_2.columns:
    counts = data_ddpm_2[column].value_counts(normalize=True, sort=False)

    # print('data_train_dist: ', counts.to_dict())
    target_marginals[column] = counts.to_dict()
print(target_marginals)

key_list = ['年龄（分段）', '性别', '家庭收入', '房屋类型', '家庭是否有车', '是否有驾照',
                                                  '每周工作天数', '工作类型', '学生类别', '家庭人数', '是否有小孩', '通勤方式',
                                                  '通勤出发时间']
weights = []
for item in try_gene:
    weight = 1.0
    for i in range(len(item)):
        # print(key_list[i], item[i])
        # print(target_marginals[key_list[i]][item[i]])
        weight = weight * target_marginals[key_list[i]][item[i]]/training_marginals[key_list[i]][item[i]]
    # print(weight)
    weights.append(weight)

avg_weight = np.mean(np.array(weights))
print('avg_weight', avg_weight)
print('max(weights)', max(weights))
print('min(weights)', min(weights))

var = 0.0
for it in weights:
    var = var + (it/avg_weight - 1)*(it/avg_weight - 1)
var = var/len(weights)

print(var)
print(1000.0/(1+var))

ess = 1000.0/(1+var)

N = 1000*(1000/ess)
print(N)


ff = open('./copula_popu_gene_nths_3_1000000.json', 'r', encoding='utf')
data = json.load(ff)
ff.close()


weights = []
for one_row in data:
    prod = 1.0
    for i, one_value in enumerate(one_row):
        prod *= target_marginals[key_list[i]][one_value]/training_marginals[key_list[i]][one_value]
    weights.append(prod)
print('max(weights): ', max(weights))
normalized_weights = np.array(weights)/np.sum(weights)
# 重采样
resampled_idx = np.random.choice(len(data), size=100000, p=normalized_weights)

resampled_data = []
for idx in resampled_idx:
    resampled_data.append(data[idx])

data_res = pd.DataFrame(resampled_data, columns=key_list)

data_res.to_csv('./resampled_data_new_100000.csv', index=False)

resam_dict = {}
for column in data_res.columns:
    counts = data_res[column].value_counts(normalize=True)
    # print(counts)
    resam_dict[column] = counts.to_dict()
# data.to_csv('resampled_data_100000.csv', index=False)
print(resam_dict)

mse1 = 0.0
mse2 = 0.0
mse3 = 0.0
for key in resam_dict.keys():
    for subkey in resam_dict[key].keys():
        mse1 += (resam_dict[key][subkey]-target_marginals[key][subkey])*(resam_dict[key][subkey]-target_marginals[key][subkey])
        mse2 += (resam_dict[key][subkey]-training_marginals[key][subkey])*(resam_dict[key][subkey]-training_marginals[key][subkey])
        mse3 += (target_marginals[key][subkey]-training_marginals[key][subkey])*(target_marginals[key][subkey]-training_marginals[key][subkey])

print(mse1/len(resam_dict.keys()))
print(mse2/len(resam_dict.keys()))
print(mse3/len(resam_dict.keys()))
