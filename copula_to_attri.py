from copula_tools import *
import json
import pandas as pd


def find_nearest(series, target):
    """返回 series 中离 target 最近的值"""
    return (series - target).abs().argmin()


f = open('./training/DDPM_cond_gene_nhts_3/voc_map.json', 'r', encoding='utf-8')
data_map = json.load(f)
f.close()
print(data_map)

voc_map = {}
for key, value in data_map.items():
    voc_map[value] = key

f1 = open('./100000_gene_nhts_3.json', 'r', encoding='utf-8')
origi_list = json.load(f1)
f1.close()
u_values = origi_list
print(u_values[:10])

for i in range(10):
    idx = i + 1
    f1 = open(f'./100000_gene_nhts_3_{idx}.json', 'r', encoding='utf-8')
    origi_list = json.load(f1)
    f1.close()
    u_values.extend(origi_list)
print(len(u_values))

f = open('./training/DDPM_cond_gene_nhts_3/training_sample_50000.json', 'r', encoding='utf')
training_data = json.load(f)
f.close()

data_train = pd.DataFrame(training_data, columns=['年龄（分段）', '性别', '家庭收入', '房屋类型', '家庭是否有车', '是否有驾照',
                                                  '每周工作天数', '工作类型', '学生类别', '家庭人数', '是否有小孩', '通勤方式',
                                                  '通勤出发时间'])
copular = CopulaScaler()
copular.fit(data_train)
print(copular.ecdf_map)
u_dist = copular.u_dist


attr_list = ['年龄（分段）', '性别', '家庭收入', '房屋类型', '家庭是否有车', '是否有驾照',
                             '每周工作天数', '工作类型', '学生类别', '家庭人数',
                             '是否有小孩', '通勤方式', '通勤出发时间']
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
for i in range(len(all_row_inv)):
    one_row = []
    for col in all_row_inv.columns:
        one_value = all_row_inv.iloc[i][col]
        one_row.append(int(one_value))
    if i < 10:
        print(one_row)
    all_row_decode.append(one_row)

ff = open('./copula_popu_gene_nths_3_1000000_new.json', 'w', encoding='utf-8')
json.dump(all_row_decode, ff, ensure_ascii=False)
ff.close()
