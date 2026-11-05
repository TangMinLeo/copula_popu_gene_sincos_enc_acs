from copula_tools import *
import json
import pandas as pd
from modules import Net_model, Diffusion
import torch


def find_nearest(series, target):
    """返回 series 中离 target 最近的值"""
    return (series - target).abs().argmin()


f = open('../popu_diffusion/models/DDPM_category7/voca_map.json', 'r', encoding='utf-8')
data_map = json.load(f)
f.close()
print(data_map)

inverse_map = {}
for k, v in data_map.items():
    inverse_map[v] = int(k)

f1 = open('./gene_pos_emb_test_retrain_new_rst_1.json', 'r', encoding='utf-8')
origi_list = json.load(f1)
f1.close()

f2 = open('./wuhan_gene_100000_samples.json', 'r')
wuhan_gene = json.load(f2)
f2.close()

u_values = []
for item in origi_list:
    # u_values.extend(item)
    u_values.append(item)
print('len(u_values): ', len(u_values))

target_marginals = {'性别': {0: 0.4748603351955307, 1: 0.5251396648044693},
             '年龄（分段）': {4: 0.034636871508379886, 8: 0.11620111731843576, 5: 0.15307262569832403,
                            9: 0.14860335195530727,
                            2: 0.12067039106145251, 7: 0.1418994413407821, 3: 0.2011173184357542,
                            6: 0.08379888268156424},
             '学历': {12: 0.1653631284916201, 16: 0.004469273743016759, 10: 0.19441340782122904,
                      15: 0.09944134078212291,
                      13: 0.2558659217877095, 14: 0.0446927374301676, 11: 0.23575418994413408},
             '人生阶段': {19: 0.5083798882681564, 22: 0.03687150837988827, 20: 0.07262569832402235,
                          18: 0.2581005586592179,
                          17: 0.12402234636871508, 21: 0.0},
             '子女年龄': {27: 0.055865921787709494, 28: 0.12290502793296089, 26: 0.12737430167597766,
                          23: 0.1441340782122905,
                          24: 0.5497206703910615, 25: 0.00},
             '是否有车': {29: 0.5106145251396648, 30: 0.4893854748603352},
             '通勤方式': {34: 0.1005586592178771, 35: 0.3139664804469274, 32: 0.11620111731843576,
                          33: 0.3575418994413408,
                          31: 0.11173184357541899},
             '酒店消费水平': {38: 0.01005586592178771, 37: 0.07932960893854749, 44: 0.28156424581005585,
                              42: 0.0011173184357541898, 43: 0.1664804469273743, 41: 0.04022346368715084,
                              40: 0.017877094972067038, 39: 0.008938547486033519, 36: 0.394413407821229},
             '餐饮消费水平': {51: 0.17206703910614526, 48: 0.09944134078212291, 49: 0.06927374301675977,
                              47: 0.06480446927374302, 45: 0.06927374301675977, 50: 0.1463687150837989,
                              52: 0.19776536312849163, 46: 0.18100558659217877},
             '手机品牌': {58: 0.10837988826815642, 57: 0.060335195530726256, 60: 0.02905027932960894,
                          53: 0.07150837988826815,
                          59: 0.2849162011173184, 54: 0.1553072625698324, 56: 0.2782122905027933,
                          55: 0.012290502793296089},
             '手机价格（分段）v2': {65: 0.07039106145251396, 61: 0.1877094972067039, 67: 0.21452513966480447,
                                  66: 0.13631284916201117,
                                  64: 0.06815642458100558, 62: 0.17094972067039105, 70: 0.08491620111731843,
                                  63: 0.024581005586592177,
                                  68: 0.024581005586592177, 69: 0.017877094972067038},
             '消费水平（分段）': {72: 0.4491620111731844, 75: 0.1441340782122905, 74: 0.26033519553072626,
                                73: 0.13072625698324022, 71: 0.01564245810055866},
             '差旅常客': {76: 0.6424581005586593, 77: 0.3575418994413408},
             '旅游距离': {79: 0.12849162011173185, 81: 0.4893854748603352, 78: 0.14748603351955308,
                          80: 0.2346368715083799},
             '出国游': {82: 0.906145251396648, 83: 0.09385474860335195}}

marginals_t = {}
for key in target_marginals.keys():
    item = target_marginals[key]
    new_item = {}
    for subkey in item.keys():
        new_item[data_map[str(subkey)]] = item[subkey]
    marginals_t[key] = new_item
# print(marginals_t)

marginals_e = {'性别': {0: 0.54613, 1: 0.45387},
              '年龄（分段）': {2: 0.14903, 3: 0.21004, 4: 0.00876, 5: 0.14727, 6: 0.12326, 7: 0.10008, 8: 0.08156,
                             9: 0.18},
              '学历': {10: 0.2075, 11: 0.18024, 12: 0.19534, 13: 0.28608, 14: 0.0299, 15: 0.10014, 16: 0.0008},
              '人生阶段': {17: 0.1632, 18: 0.28828, 19: 0.5129, 20: 0.0257, 21: 0.0, 22: 0.00992},
              '子女年龄': {23: 0.16094, 24: 0.32897, 25: 0.21137, 26: 0.13767, 27: 0.06057, 28: 0.10048},
              '是否有车': {29: 0.49701, 30: 0.50299},
              '通勤方式': {31: 0.13548, 32: 0.14387, 33: 0.34189, 34: 0.13306, 35: 0.2457},
              '酒店消费水平': {36: 0.34301, 37: 0.18691, 38: 0.00777, 39: 0.00345, 40: 0.00973, 41: 0.0445, 42: 0.00156,
                               43: 0.12075, 44: 0.28232},
              '餐饮消费水平': {45: 0.0579, 46: 0.12105, 47: 0.05453, 48: 0.12927, 49: 0.09252, 50: 0.19583, 51: 0.13932,
                               52: 0.20958},
              '手机品牌': {53: 0.07519, 54: 0.14549, 55: 0.0002, 56: 0.23624, 57: 0.08171, 58: 0.10003, 59: 0.35835,
                           60: 0.00279},
              '手机价格（分段）v2': {61: 0.19764, 62: 0.16355, 63: 0.01583, 64: 0.11011, 65: 0.06585, 66: 0.13823,
                                   67: 0.23023, 68: 0.00659, 69: 0.00202, 70: 0.06995},
              '消费水平（分段）': {71: 0.00714, 72: 0.45792, 73: 0.07511, 74: 0.22348, 75: 0.23635},
              '差旅常客': {76: 0.72842, 77: 0.27158},
              '旅游距离': {78: 0.11224, 79: 0.1189, 80: 0.24662, 81: 0.52224},
              '出国游': {82: 0.97616, 83: 0.02384}}


copular_t = CopulaScaler()
copular_t.fit_from_marginals(target_marginals)
mapping_t = copular_t.reverse_mapping

# print(copular_t.ecdf_map)
u_dist_t = copular_t.u_dist
# print(u_dist_t)

copular_e = CopulaScaler()
copular_e.fit_from_marginals(marginals_e)
mapping_e = copular_e.reverse_mapping

# print(copular_e.ecdf_map)
u_dist_e = copular_e.u_dist
# print(u_dist_e)

attr_list = ['性别', '年龄（分段）', '学历', '人生阶段', '子女年龄', '是否有车', '通勤方式', '酒店消费水平',
             '餐饮消费水平', '手机品牌', '手机价格（分段）v2', '消费水平（分段）', '差旅常客', '旅游距离', '出国游']

all_row = []
all_row_wo_decode = []
for item in u_values:
    one_row = []
    one_row_wo_decode = []
    for i, one_value in enumerate(item):
        key = attr_list[i]
        series = pd.Series(u_dist_e[key])
        idx = find_nearest(series, one_value)
        one_row.append(data_map[str(int(mapping_e[key][int(idx)]))])
        one_row_wo_decode.append(int(mapping_e[key][int(idx)]))

    all_row.append(one_row)
    all_row_wo_decode.append(one_row_wo_decode)


data_train = pd.read_excel(r'D:\我的桌面\社区调查\combine_all.xlsx')
data_train_unique = data_train.drop_duplicates()
print(len(data_train_unique), len(data_train))

# for column in data_train.columns:
#     counts = data_train[column].value_counts(normalize=True)
#     print('data_train_dist: ', counts)

train_data = data_train.values.tolist()
train_tupe = []
for item in train_data:
    one_tuple = []
    for i, one_value in enumerate(item):
        tem = ':'.join([attr_list[i], one_value])
        idx = int(inverse_map[tem])
        one_tuple.append(idx)

    train_tupe.append(tuple(one_tuple))

all_row_tup = []
for one_item in all_row_wo_decode:
    # tup = []
    # for one_value in one_item:
    #     tup.append(one_value.split(':')[1])

    tup = tuple(one_item)
    all_row_tup.append(tup)

origi_tup = []
for key in wuhan_gene.keys():
    origi_tup.append(tuple(wuhan_gene[key]))

print('train_tupe[:10]: ', train_tupe[:10])
print('all_row_tup[:10]: ', all_row_tup[:10])
print('origi_tup[:10]: ', origi_tup[:10])

gene_sample_in_train = []
count_not_in = 0
gene_in = 0
for row in all_row_tup:
    if row not in train_tupe:
        count_not_in += 1
    if row in train_tupe:
        gene_in += 1
        gene_sample_in_train.append(row)
gene_sample_in_train_s = pd.DataFrame(np.array(gene_sample_in_train), columns=attr_list)
de_duplicate = gene_sample_in_train_s.drop_duplicates()
print('len(de_duplicate): ', len(de_duplicate))

print(gene_in, gene_in/len(all_row_tup))
print(count_not_in/len(all_row))
print(count_not_in, len(all_row))

train_sample_in_gene = []
count_in = 0
for row in train_tupe:
    if row in all_row_tup:
        count_in += 1
        train_sample_in_gene.append(row)
train_sample_in_gene_s = pd.DataFrame(np.array(train_sample_in_gene), columns=attr_list)
de_duplicate_1 = train_sample_in_gene_s.drop_duplicates()
print('len(de_duplicate_1): ', len(de_duplicate_1))

print(count_in/len(train_tupe))
print(count_in, len(train_tupe))

splited_all_rows = []
for one_row in all_row:
    splited_row = []
    for one_value in one_row:
        splited_row.append(one_value.split(':')[1])
    splited_all_rows.append(splited_row)


data = pd.DataFrame(splited_all_rows, columns=attr_list)
data.to_csv('popu_data_generated_100000.csv', index=False)

all_row_unique = data.drop_duplicates()
print(len(all_row_unique), len(all_row))

train_sample_in_orig = 0
for one_row in train_tupe:
    if one_row in origi_tup:

        train_sample_in_orig += 1
print('train_sample_in_orig: ', train_sample_in_orig)

ori_sample_in_train = 0
for one_row in origi_tup:
    if one_row in train_tupe:
        ori_sample_in_train += 1
print('ori_sample_in_train: ', ori_sample_in_train)

# marginals_e_update = {}
# for column in data.columns:
#     counts = data[column].value_counts(normalize=True)
#     # print(counts)
#     marginals_e_update[column] = counts.to_dict()
# print(marginals_e_update)
#
# weights = []
# for one_row in all_row:
#     prod = 1.0
#     for i, one_value in enumerate(one_row):
#         prod *= marginals_t[attr_list[i]][one_value]/marginals_e_update[attr_list[i]][one_value]
#     weights.append(prod)
# print('max(weights): ', max(weights))
# normalized_weights = np.array(weights)/np.sum(weights)
# # 重采样
# resampled_idx = np.random.choice(len(all_row), size=100000, p=normalized_weights)
#
# resampled_data = []
# for idx in resampled_idx:
#     resampled_data.append(all_row[idx])
#
# data = pd.DataFrame(resampled_data, columns=attr_list)
# for column in data.columns:
#     counts = data[column].value_counts(normalize=True)
#     print(counts)
# data.to_csv('resampled_data_100000.csv', index=False)
