import csv
import pandas as pd
import json
from copula_tools import *


# def find_nearest(series, target):
#     """返回 series 中离 target 最近的值"""
#     return (series - target).abs().argmin()
#
#
# f = open('./training/DDPM_cond_gene_nhts_3/voc_map.json', 'r', encoding='utf-8')
# data_map = json.load(f)
# f.close()
# print(data_map)
#
# voc_map = {}
# for key, value in data_map.items():
#     voc_map[value] = key
#
# f1 = open('./100000_gene_nhts_3_2.json', 'r', encoding='utf-8')
# origi_list = json.load(f1)
# f1.close()
# u_values = origi_list
# print(len(u_values))
# print(u_values[:10])
#
#
# f = open('./training/DDPM_cond_gene_nhts_3/training_sample_50000.json', 'r', encoding='utf')
# training_data = json.load(f)
# f.close()
# print('len(training_data): ', len(training_data))
#
# data_train = pd.DataFrame(training_data, columns=['年龄（分段）', '性别', '家庭收入', '房屋类型', '家庭是否有车', '是否有驾照',
#                                                   '每周工作天数', '工作类型', '学生类别', '家庭人数', '是否有小孩', '通勤方式',
#                                                   '通勤出发时间'])
# copular = CopulaScaler()
# copular.fit(data_train)
# print(copular.ecdf_map)
# u_dist = copular.u_dist
#
#
# attr_list = ['年龄（分段）', '性别', '家庭收入', '房屋类型', '家庭是否有车', '是否有驾照',
#                              '每周工作天数', '工作类型', '学生类别', '家庭人数',
#                              '是否有小孩', '通勤方式', '通勤出发时间']
# data = pd.DataFrame(u_values, columns=attr_list)
#
# all_row = []
# for i in range(len(data)):
#     one_row = []
#     for col in data.columns:
#         key = col
#         series = pd.Series(u_dist[key])
#         one_value = data.iloc[i][col]
#         idx = find_nearest(series, one_value)
#         one_row.append(series[idx])
#     if i < 10:
#         print(one_row)
#     all_row.append(one_row)
#
# all_row_df = pd.DataFrame(all_row, columns=attr_list)
# all_row_inv = copular.inverse_transform(all_row_df)
#
#
# all_row_decode = []
# for i in range(len(all_row_inv)):
#     one_row = []
#     for col in all_row_inv.columns:
#         one_value = all_row_inv.iloc[i][col]
#         one_row.append(int(one_value))
#     if i < 10:
#         print(one_row)
#     all_row_decode.append(one_row)
#
# ff = open('./copula_popu_gene_nths_5_100000_2.json', 'w', encoding='utf-8')
# json.dump(all_row_decode, ff, ensure_ascii=False)
# ff.close()


ff = open('./copula_popu_gene_nths_3_1066319.json', 'r', encoding='utf')
json_data = json.load(ff)
ff.close()

# ff = open('./copula_popu_gene_nths_3_100000.json', 'r', encoding='utf')
# json_data_1 = json.load(ff)
# ff.close()

tuple_gene = []
for item in json_data:
    tuple_gene.append(tuple(item))

# for i in range(66319):
#     tuple_gene.append(tuple(json_data_1[i]))

print('len(tuple_gene): ', len(tuple_gene))
print(tuple_gene[:10])

file_name = r'D:\Download\Kim and Bansal (2023)_TR-C\data_NHTSint_indiv.csv'
f = open(file_name, 'r', encoding='utf-8')
reader = csv.reader(f)

f0 = open('./training/DDPM_cond_gene_nhts_3/voc_map.json', 'r', encoding='utf-8')
voca_map = json.load(f0)
f0.close()

# vocabulary = {}
# for key, value in voca_map.items():
#     vocabulary[value] = key

count = 0
dict_dims = {}
idx_map = {}

for row in reader:
    if count == 0:
        col_count = 0
        for item in row:
            dict_dims[item] = []
            idx_map[col_count] = item
            col_count += 1
        count = count + 1
    else:
        for j in range(len(row)):
            dict_dims[idx_map[j]].append(row[j])
        count = count + 1

print(count)
f.close()

all_tuple = []
for i in range(count-1):
    one_sample = []
    for key in dict_dims.keys():
        one_sample.append(int(voca_map[f'{key}:{dict_dims[key][i]}']))
    all_tuple.append(tuple(one_sample))
print(len(all_tuple))
print(all_tuple[:10])

cou = 0
for item in tuple_gene:
    if item in all_tuple:
        cou = cou + 1
print('precision: ', cou/len(tuple_gene))

coun = 0
for item in all_tuple:
    if item in tuple_gene:
        coun = coun + 1
print('recall: ', coun/len(all_tuple))
