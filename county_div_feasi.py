import json
import pandas as pd
from importance_resampling import importance_resamp

f = open('./voca_map.json', 'r', encoding='utf-8')
data_map = json.load(f)
f.close()
# print(data_map)

file_name = './resampled_from_gene_copula_ddpm6_decoded_1306799_new.csv'
print(file_name)
df = pd.read_csv(file_name, dtype='int')
df = df[['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC']]
print(df.head())

gene_data = df.values.tolist()
print('gene_data[:5]: ', gene_data[:5])
print(len(gene_data))

idxs = ['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC']

# gene_decoded = []
# for item in gene_data:
#     one_decoded = []
#     for i in range(len(item)):
#         one_decoded.append(data_map[f'{idxs[i]}:{item[i]}'])
#     gene_decoded.append(one_decoded)
# print(gene_decoded[:5])
# print('len(gene_decoded): ', len(gene_decoded))

f = open('./mapped_tx.json', 'r')
target_data = json.load(f)
f.close()
print('len(target_data): ', len(target_data))
print('target_data[:5] ', target_data[:5])
data_tar = pd.DataFrame(target_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])

f = open('./mapped_tn.json', 'r')
training_data = json.load(f)
f.close()
print('len(training_data): ', len(training_data))
print('training_data[:5] ', training_data[:5])
data_train = pd.DataFrame(training_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])

source_set = set(tuple(i) for i in data_train.to_numpy())
target_set = set(tuple(i) for i in data_tar.to_numpy())
synthetic_set = set(tuple(i) for i in gene_data)
# synthetic_set = set(tuple(i) for i in df.to_numpy())


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
for item in gene_data:
    synthetic_list.append(tuple(item))

print('begin calculating recall and precison')
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
