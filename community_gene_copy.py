from copula_tools import *
import json
import pandas as pd
from modules import Net_model, Diffusion
import torch
from tqdm import tqdm


def find_nearest(series, target):
    """返回 series 中离 target 最近的值"""
    return (series - target).abs().argmin()


f = open('./voca_map.json', 'r', encoding='utf-8')
data_map = json.load(f)
f.close()
print(data_map)

voc_map = {}
for key, value in data_map.items():
    voc_map[value] = key

f1 = open('./100000_gene_acs_2.json', 'r', encoding='utf-8')
origi_list = json.load(f1)
f1.close()
u_values = origi_list
print(len(u_values))
print(u_values[:10])


f = open('./mapped.json', 'r', encoding='utf')
training_data = json.load(f)
f.close()
print(len(training_data))

data_train = pd.DataFrame(training_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])
copular = CopulaScaler()
copular.fit(data_train)
print(copular.ecdf_map)
u_dist = copular.u_dist


attr_list = ['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC']
data = pd.DataFrame(u_values, columns=attr_list)

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


all_row_decode = []
all_encoded = []
for i in range(len(all_row_inv)):
    one_row = []
    one_row_enc = []
    for col in all_row_inv.columns:
        one_value = all_row_inv.iloc[i][col]
        one_row_enc.append(int(one_value))
        one_row.append(voc_map[int(one_value)])
    if i < 10:
        print(one_row)
        print(one_row_enc)
    all_row_decode.append(one_row)
    all_encoded.append(one_row_enc)

ff = open('./copula_popu_gene_acs_2_100000.json', 'w', encoding='utf-8')
json.dump(all_row_decode, ff, ensure_ascii=False)
ff.close()

f2 = open('./copula_popu_gene_acs_voc_enc_2_100000.json', 'w', encoding='utf-8')
json.dump(all_encoded, f2, ensure_ascii=False)
f2.close()
print(len(all_encoded))
