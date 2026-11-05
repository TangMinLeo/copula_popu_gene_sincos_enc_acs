from copula_tools import *
import json
import pandas as pd
from importance_resampling import importance_resamp


def find_nearest(series, target):
    """返回 series 中离 target 最近的值"""
    return (series - target).abs().argmin()


f = open('./mapped_ca.json', 'r')
target_data = json.load(f)
f.close()
print('len(target_data): ', len(target_data))

f = open('./mapped.json', 'r', encoding='utf')
training_data = json.load(f)
f.close()
print('len(training_data): ', len(training_data))
data_train = pd.DataFrame(training_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])

copular = CopulaScaler()
copular.fit(data_train)
print(copular.ecdf_map)
u_dist = copular.u_dist

gene_from_md_all = []
for i in range(19):
    file_name = f'100000_gene_acs_{i}.json'
    f = open(file_name, 'r')
    temp = json.load(f)
    f.close()
    gene_from_md_all.extend(temp)
print('len(gene_from_md_all): ',len(gene_from_md_all))
print(gene_from_md_all[:10])
gene_from_md_all = gene_from_md_all[:len(target_data)]
print('len(gene_aligned_tar): ', len(gene_from_md_all))

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




f = open('gene_copula_2_decoded_1853429.json', 'r')
saved = json.load(f)
f.close()

source_set = set(tuple(i) for i in all_encoded)
target_set = set(tuple(i) for i in saved)

intersect = source_set.intersection(target_set)
print(len(intersect))

