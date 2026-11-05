import json
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from copula_tools import *

mpl.rcParams['font.family'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

f = open('../popu_diffusion/models/DDPM_category7/voca_map.json', 'r', encoding='utf-8')
dict_mapping = json.load(f)
f.close()

print(dict_mapping)
descriptor = list(dict_mapping.values())
all_attributes = []
distributions_vals = {}
for item in descriptor:
    all_attributes.append(item.split(':')[0])
    if distributions_vals.get(item.split(':')[0]) is None:
        distributions_vals[item.split(':')[0]] = []
        distributions_vals[item.split(':')[0]].append(item.split(':')[1])
    else:
        distributions_vals[item.split(':')[0]].append(item.split(':')[1])


attributes = list(OrderedDict.fromkeys(all_attributes))
print(attributes)
print(distributions_vals)

dict_marginals = {}
for attribute in attributes:
    dict_marginals[attribute] = {}
    for it in distributions_vals[attribute]:
        dict_marginals[attribute][':'.join([attribute, it])] = 0

file_json = open(f"./wuhan_gene_100000_samples.json", "r")
data = json.load(file_json)
file_json.close()

for key in data.keys():
    sample = data[key]
    for i in range(len(sample)):
        temp = str(sample[i])
        dict_marginals[attributes[i]][dict_mapping[temp]] = dict_marginals[attributes[i]][dict_mapping[temp]] + 1

print(dict_marginals)

i = 0
for key in dict_marginals.keys():
    for subkey in dict_marginals[key].keys():
        dict_marginals[key][subkey] = dict_marginals[key][subkey]/len(data)
    print(dict_marginals[key])

    # fig = plt.figure(i)
    # plt.bar(list(range(len(dict_marginals[key]))), list(dict_marginals[key].values()))
    # plt.xticks(list(range(len(dict_marginals[key]))), list(dict_marginals[key].keys()), rotation=30)
    # # for j in range(len(dict_marginals[key])):
    # #     plt.text(j, value + 1, str(value), ha='center', va='bottom')
    #
    # plt.title(key)
    # plt.show()
    # plt.close(fig)
    # i = i + 1

file_json = open(f"./wuhan_gene_100000_samples.json", "r")
    # file_json = open(json_file_path, "r")

json_data = json.load(file_json)
file_json.close()

data = pd.DataFrame.from_dict(json_data, orient='index', columns=['性别', '年龄（分段）', '学历', '人生阶段', '子女年龄',
                                                                      '是否有车', '通勤方式', '酒店消费水平', '餐饮消费水平',
                                                                      '手机品牌', '手机价格（分段）v2', '消费水平（分段）',
                                                                      '差旅常客', '旅游距离', '出国游'])
# X = data.head(100)
# print(data.columns)
print(data.head(100))

copula = CopulaScaler()
copula.fit(data)

transformed_data = copula.transform(data)

print(transformed_data.head(100))
