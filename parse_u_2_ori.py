from train_copula_cond import find_nearest
import pandas as pd
import json
from copula_tools import *

save_json_file = './gene_sincos_enc_nhts_0_100000.json'
f = open(save_json_file, 'r')
sampled_items = json.load(f)
f.close()

f = open('./training_sample_1.json', 'r')
training_samples = json.load(f)
f.close()

f_voc = open('voc_map.json', 'r', encoding='utf')
voc_map = json.load(f_voc)
f_voc.close()
print(voc_map)
inv_map = {}
for key, value in voc_map.items():
    inv_map[value] = key

training_data = pd.DataFrame(training_samples, columns=['年龄（分段）', '性别', '家庭收入', '房屋类型', '家庭是否有车', '是否有驾照',
                                                                      '每周工作天数', '工作类型', '学生类别', '家庭人数',
                                                                      '是否有小孩', '通勤方式', '通勤出发时间'])
copula = CopulaScaler()
copula.fit(training_data)

data = pd.DataFrame(sampled_items, columns=['年龄（分段）', '性别', '家庭收入', '房屋类型', '家庭是否有车', '是否有驾照',
                                                                      '每周工作天数', '工作类型', '学生类别', '家庭人数',
                                                                      '是否有小孩', '通勤方式', '通勤出发时间'])

u_dist = copula.u_dist
# mapping1 = copula.reverse_mapping
all_row = []
for i in range(len(data)):
    one_row = []
    for col in data.columns:
        key = col
        series = pd.Series(u_dist[key])
        one_value = data.iloc[i][col]
        idx = find_nearest(series, one_value)
        one_row.append(series[idx])
        # one_row.append(data_map[str(int(mapping1[key][int(idx)]))])
    # print(one_row)
    all_row.append(one_row)

total_row = pd.DataFrame(all_row, columns=['年龄（分段）', '性别', '家庭收入', '房屋类型', '家庭是否有车', '是否有驾照',
                                                              '每周工作天数', '工作类型', '学生类别', '家庭人数',
                                                              '是否有小孩', '通勤方式', '通勤出发时间'])

inversed_cdf = copula.inverse_transform(total_row)

sample_out = inversed_cdf.values.tolist()
gene_rep = []
for j in range(len(sample_out)):
    tem = []
    for item in sample_out[j]:
        tem.append(inv_map[int(item)])
    # gene_rep.append(tem)
    if j % 1000 == 0:
        print(tem)

    gene_rep.extend(tem)

ff = open('u_2_ori.json', 'w', encoding='utf')
json.dump(gene_rep, ff, ensure_ascii=False)
ff.close()
