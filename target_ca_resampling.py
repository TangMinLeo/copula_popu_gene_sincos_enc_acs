from copula_tools import *
import json
import pandas as pd
from modules import Net_model, Diffusion
import torch
from tqdm import tqdm
from importance_resampling import importance_resamp
import itertools


def find_nearest(series, target):
    """返回 series 中离 target 最近的值"""
    return (series - target).abs().argmin()

def srmse(data1, data2):
    """ Compute Standardized Root Mean Squared Error between two datasets.

    Reference: https://www.researchgate.net/publication/282815687_A_Bayesian_network_approach_for_population_synthesis
    """
    columns = list(data1.columns.values)
    # Relative frequency
    data1_f = data1.value_counts(normalize=True, dropna=False)
    data2_f = data2.value_counts(normalize=True, dropna=False)
    # Total numbers of categories
    Mi = [data1_f.index.get_level_values(l).union(data2_f.index.get_level_values(l)).unique().size for l in range(len(columns))]
    M = np.prod(Mi)
    # SRMSE
    SRMSE = ((data1_f.subtract(data2_f, fill_value=0)**2) * M).sum()**(.5)
    return SRMSE

f = open('./voca_map.json', 'r', encoding='utf-8')
data_map = json.load(f)
f.close()
print(data_map)

voc_map = {}
for key, value in data_map.items():
    voc_map[value] = key

f = open('./mapped_tx.json', 'r')
target_data = json.load(f)
f.close()
print('len(target_data): ', len(target_data))
data_tar = pd.DataFrame(target_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])
# print(target_data[:10])
print(data_tar.head(10))

gene_from_md_all = []
for i in range(1, 15):
    file_name = f'100000_gene_acs_6_state_puma_{i}.json'
    f = open(file_name, 'r')
    temp = json.load(f)
    f.close()
    gene_from_md_all.extend(temp)
print('len(gene_from_md_all): ',len(gene_from_md_all))
print(gene_from_md_all[:10])
gene_from_md_all = gene_from_md_all[:len(target_data)]
print('len(gene_aligned_tar): ', len(gene_from_md_all))

# f = open('./50000_gene_acs_3_county.json', 'r')
# gene_data_0 = json.load(f)
# f.close()
# print('len(gene_data): ', len(gene_data_0))

# f = open('./50000_gene_acs_3_county_1.json', 'r')
# gene_data_1 = json.load(f)
# f.close()
# print('len(gene_data): ', len(gene_data_1))

# gene_data = []
# gene_data.extend(gene_data_0)
# gene_data.extend(gene_data_1[:(len(target_data)-len(gene_data_0))])
# print('len(gene_data): ', len(gene_data))

f = open('./mapped_tn.json', 'r')
training_data = json.load(f)
f.close()
print('len(training_data): ', len(training_data))
# print(training_data[:10])

data_train = pd.DataFrame(training_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])
print(data_train.head(10))


copular = CopulaScaler()
copular.fit(data_train)
print(copular.ecdf_map)
u_dist = copular.u_dist


attr_list = ['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC']
data = pd.DataFrame(gene_from_md_all, columns=attr_list)

all_row = []
for i in range(len(data)):
    one_row = []
    for col in data.columns:
        key = col
        series = pd.Series(u_dist[key])
        one_value = data.iloc[i][col]
        idx = find_nearest(series, one_value)
        one_row.append(series[idx])
    if i < 10:
        print(one_row)
    all_row.append(one_row)

all_row_df = pd.DataFrame(all_row, columns=attr_list)
all_row_inv = copular.inverse_transform(all_row_df)
print(all_row_inv.head(10))
# all_row_inv.astype(int)

# all_row_decode = []
all_encoded = []
for i in range(len(all_row_inv)):
    # one_row = []
    one_row_enc = []
    for col in all_row_inv.columns:
        one_value = all_row_inv.iloc[i][col]
        one_row_enc.append(int(one_value))
        # one_row.append(voc_map[int(one_value)])
    if i < 10:
        # print(one_row)
        print(one_row_enc)
    # all_row_decode.append(one_row)
    all_encoded.append(one_row_enc)

print('len(all_encoded): ', len(all_encoded))

f = open('gene_copula_ddpm6_decoded_1306799.json', 'w')
json.dump(all_encoded, f)
f.close()

# f = open('gene_copula_2_decoded.json', 'r')
# all_encoded = json.load(f)
# f.close()

resampled = importance_resamp(data_train, data_tar, all_encoded)
print('len(resampled): ', len(resampled))

resampled.to_csv('./resampled_from_gene_copula_ddpm6_decoded_1306799_new.csv', index=False)

# data_gene = pd.DataFrame(all_encoded, columns=attr_list)

srmse_dict = {}
columns = data_train.columns

for i in range(1, 6):
    tuples = list(itertools.combinations(columns, i))  # No repeated elements
    SRMSE = 0 
    for tuple in tuples:
        SRMSE += srmse(data_tar.drop(list(columns.difference(tuple)), axis=1), resampled.drop(list(columns.difference(tuple)), axis=1))
    SRMSE /= len(tuples)
    srmse_dict["SRMSE "+str(i)] = SRMSE
# srmse_dict[model]["Zeros"] = sampling_zeros(source, target, df)
print('target and resampling ', srmse_dict)
