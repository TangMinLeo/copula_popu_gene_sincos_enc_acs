import json
import pandas as pd
from importance_resampling import importance_resamp

f = open('./gene_copula_2_decoded_1853429.json', 'r')
all_encoded = json.load(f)
f.close()
print('len(all_encoded): ', len(all_encoded))

f = open('./mapped_ca.json', 'r')
target_data = json.load(f)
f.close()
print('len(target_data): ', len(target_data))
data_tar = pd.DataFrame(target_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])

# f = open('./puma_mapped_montgomery.json', 'r')
# target_data = json.load(f)
# f.close()
# print('len(target_data): ', len(target_data))
# data_tar = pd.DataFrame(target_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])

f = open('./mapped.json', 'r', encoding='utf')
training_data = json.load(f)
f.close()
print('len(training_data): ', len(training_data))
data_train = pd.DataFrame(training_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])

# 这里生成的样本还没有目标样本多，因此下面的两行代码注释掉
# data_gene = all_encoded[:len(target_data)]
# resampled = importance_resamp(data_train, data_tar, data_gene)
resampled = importance_resamp(data_train, data_tar, all_encoded)
resampled.to_csv('./resampled_from_gene_copula_2_decoded_1853429.csv', index=False)


print('len(resampled): ', len(resampled))

source_set = set(tuple(i) for i in data_train.to_numpy())
target_set = set(tuple(i) for i in data_tar.to_numpy())
synthetic_set = set(tuple(i) for i in resampled.to_numpy())

intersect = synthetic_set.intersection(target_set)
zeros = intersect - source_set
structural_zeros = synthetic_set - target_set

print('len(structural_zeros): ', len(structural_zeros))
print('sampling zeros: ', len(zeros))

# source_list = []
# for item in training_data:
#     source_list.append(tuple(item))

target_list = []
for item in target_data:
    target_list.append(tuple(item))
    
synthetic_list = []
for item in resampled.to_numpy():
    synthetic_list.append(tuple(item))

count_recall = 0
for item in target_list:
    if item in synthetic_list:
        count_recall += 1
print('recall: ', count_recall/len(target_list))

count_precison = 0
for item in synthetic_list:
    if item in target_list:
        count_precison += 1
print('precison: ', count_precison/len(synthetic_list))
