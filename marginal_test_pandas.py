import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd

mpl.rcParams['font.family'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

file = open('./copula_popu_gene_nths_3_1066319.json', 'r')
dict_gene = json.load(file)
file.close()

file1 = open('./training/DDPM_cond_gene_nhts_3/voc_map.json', 'r')
dict_voca = json.load(file1)
file1.close()

fig_all = plt.figure(105, figsize=(25, 15))

# age_mapping = {'[5,10)': 0, '[10,15)': 1, '[15,20)': 2, '[20,25)': 3, '[25,30)': 4, '[30,35)': 5, '[35,40)': 6,
#                '[40,45)': 7, '[45,50)': 8, '[50,55)': 9, '[55,60)': 10, '[60,65)': 11, '[65,70)': 12, '[70,75)': 13,
#                '[75,80)': 14, '[80,85)': 15, '[85,90]': 16}
age_list = ['[5,10)', '[10,15)', '[15,20)', '[20,25)', '[25,30)', '[30,35)', '[35,40)', '[40,45)', '[45,50)', '[50,55)',
            '[55,60)', '[60,65)', '[65,70)', '[70,75)', '[75,80)', '[80,85)', '[85,90]']

ii = 0
new_voca = {}
attr_names = []
# "Age:[30,35)": 0
for key in dict_voca.keys():
    new_voca[dict_voca[key]] = key
    attr = key.split(':')[0]
    attr_names.append(attr)
# attr_names = list(set(attr_names))
de_dupi = []
for one_it in attr_names:
    if one_it not in de_dupi:
        de_dupi.append(one_it)

print(attr_names)
print(de_dupi)

dict_dims_gene = {}
idx_map_gene = {}

rmv_dup = []

attr_count = 0
for attr in de_dupi:
    dict_dims_gene[attr] = []
    idx_map_gene[attr_count] = attr
    attr_count += 1

count = 0
for item in dict_gene:
    count = count + 1
    col_num = 0
    to_tuple = tuple(item)
    rmv_dup.append(to_tuple)

    for sub_item in item:
        temp = new_voca[sub_item]
        tem = temp.split(':')[1]
        dict_dims_gene[idx_map_gene[col_num]].append(tem)
        col_num = col_num + 1


# for key in dict_gene.keys():
#     if key == '1':
#         col_num = 0
#         for item in dict_gene[key]:
#             idx_map_gene[col_num] = item
#             dict_dims_gene[item] = []
#             col_num = col_num + 1
#     else:
#         col_num = 0
#         count = count + 1
#         to_tuple = tuple(dict_gene[key])
#         rmv_dup.append(to_tuple)
#
#         for item in dict_gene[key]:
#             temp = new_voca[item]
#             tem = temp.split(':')[1]
#             dict_dims_gene[idx_map_gene[col_num]].append(tem)
#             col_num = col_num + 1

print(count)
print(dict_dims_gene["Age"][:10])
print(len(set(rmv_dup)))

# pd_data_gene = pd.DataFrame(data=dict_dims_gene)
# for col in pd_data_gene.columns:
#     print(pd_data_gene[col].value_counts(normalize=True))
#
# for i in range(len(de_dupi)):
#     for j in range(i + 1, len(de_dupi)):
#         joint_distribution = pd.crosstab(pd_data_gene[de_dupi[i]], pd_data_gene[de_dupi[j]], margins=True)
#         joint_distribution_porb = joint_distribution.div(joint_distribution.loc["All", "All"])
#         print(joint_distribution_porb)

csv_file = r'D:\Download\Kim and Bansal (2023)_TR-C\data_NHTSint_indiv.csv'
dict_dims = {}
idx_map = {}

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    count = 0
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
# print(dict_dims)


def descriptive_statistic(dict_dims):
    unique_dims = {}
    total_ans = 0
    for key in dict_dims.keys():
        # print(key, len(dict_dims[key]))
        unique_dims[key] = list(set(dict_dims[key]))
        total_ans = total_ans + len(unique_dims[key])
        # print(unique_dims[key])

    print(total_ans)

    count_dict_total = {}
    for key in dict_dims.keys():
        count_dict_total[key] = {}
        for item in unique_dims[key]:
            count_dict_total[key][item] = 0

        total_num = 0
        for ans in dict_dims[key]:
            count_dict_total[key][ans] = count_dict_total[key][ans] + 1
            total_num = total_num + 1
        print(total_num)
        # print(key, count_dict_total[key])

        for item in unique_dims[key]:
            count_dict_total[key][item] = count_dict_total[key][item] / total_num
        print(key, count_dict_total[key])

    bi_marginal_count = {}

    keys = list(dict_dims.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            bi_marginal_count[f'{keys[i]}:{keys[j]}'] = {}
            for item1 in unique_dims[keys[i]]:
                for item2 in unique_dims[keys[j]]:
                    bi_marginal_count[f'{keys[i]}:{keys[j]}'][f'{item1}:{item2}'] = 0

            total_num = 0
            for k in range(len(dict_dims[keys[i]])):
                bi_marginal_count[f'{keys[i]}:{keys[j]}'][f'{dict_dims[keys[i]][k]}:{dict_dims[keys[j]][k]}'] = (
                        bi_marginal_count[f'{keys[i]}:{keys[j]}'][
                            f'{dict_dims[keys[i]][k]}:{dict_dims[keys[j]][k]}'] + 1)
                total_num = total_num + 1
            print(total_num)
            # print(f'{keys[i]}:{keys[j]}', bi_marginal_count[f'{keys[i]}:{keys[j]}'])

            for item1 in unique_dims[keys[i]]:
                for item2 in unique_dims[keys[j]]:
                    bi_marginal_count[f'{keys[i]}:{keys[j]}'][f'{item1}:{item2}'] = (
                            bi_marginal_count[f'{keys[i]}:{keys[j]}'][f'{item1}:{item2}'] / total_num)

            print(f'{keys[i]}:{keys[j]}', bi_marginal_count[f'{keys[i]}:{keys[j]}'])

    return count_dict_total, bi_marginal_count


marginal_gene, bi_marginal_gene = descriptive_statistic(dict_dims_gene)
marginal_data, bi_marginal_data = descriptive_statistic(dict_dims)


barture = ''
bargene = ''

for key in marginal_data.keys():
    marg1 = marginal_data[key]
    marg2 = marginal_gene[key]
    prob1 = []
    prob2 = []
    subkey_list = []

    if key == 'Age':
        subkey_list = age_list
        # for subkey in marg1.keys():
        #     subkey_list.append(age_mapping[subkey])
    else:
        for subkey in marg1.keys():
            subkey_list.append(subkey)

        subkey_list.sort()

    for subkey in subkey_list:
        prob1.append(marg1[subkey])
        prob2.append(marg2[subkey])

    axes_all = fig_all.add_subplot(3, 5, ii + 1)

    x = np.arange(len(prob1))

    bar_true = axes_all.bar(x, prob1, 0.2)
    bar_gene = axes_all.bar(x + 0.2, prob2, 0.2)
    if key == 'Age':
        axes_all.set_xticks(list(x), subkey_list, rotation=50)
    else:
        axes_all.set_xticks(list(x), subkey_list, rotation=35)
    axes_all.set_title(key)
    # axes_all.legend([bar_true[0], bar_gene[0]], ['ture data', 'gene data'], loc='upper right')
    axes_all.set_ylabel('Probability')

    # plt.figure(ii)
    # plt.bar(x, prob1, 0.2)
    # plt.bar(x + 0.2, prob2, 0.2)
    #
    # plt.xticks(list(x), subkey_list, rotation=35)
    # # plt.xlabel(ke)
    # plt.title(key + '(蓝色：真实数据，橙色：生成数据)')
    # plt.ylabel("概率")
    # plt.savefig(rf"{key}" + '.png')
    # plt.close(ii)
    # plt.show()
    barture = bar_true[0]
    bargene = bar_gene[0]
    ii = ii + 1

fig_all.legend([barture, bargene], ['ture data', 'gene data'], bbox_to_anchor=(0.8, 0.3), fontsize=15)
fig_all.suptitle('Attribute-wise marginal distribution (SRMSE = 0.014)', fontsize=20)
fig_all.tight_layout()
fig_all.show()

# plt.legend([barture, bargene], ['ture data', 'gene data'], loc='lower right')
# plt.title('Attribute-wise marginal distribution (SRMSE = 0.014)')
# plt.show()

concat_prob1 = []
concat_prob2 = []
for key in marginal_data.keys():
    marg1 = marginal_data[key]
    marg2 = marginal_gene[key]
    prob1 = []
    prob2 = []
    subkey_list = []

    for subkey in marg1.keys():
        subkey_list.append(subkey)
        prob1.append(marg1[subkey])
        prob2.append(marg2[subkey])

    concat_prob1.extend(prob1)
    concat_prob2.extend(prob2)

concat_prob1 = np.array(concat_prob1)
concat_prob2 = np.array(concat_prob2)

rmse_mar = np.linalg.norm(concat_prob1 - concat_prob2)/np.sqrt(len(concat_prob1))
ybar_mar = concat_prob1.mean()
srmse_mar = rmse_mar/ybar_mar

print("srmse_mar", srmse_mar)


concat_bi_prob1 = []
concat_bi_prob2 = []
for key in bi_marginal_data.keys():
    bi_marg1 = bi_marginal_data[key]
    bi_marg2 = bi_marginal_gene[key]
    bi_prob1 = []
    bi_prob2 = []
    subkey_list = []

    for subkey in bi_marg1.keys():
        subkey_list.append(subkey)
        bi_prob1.append(bi_marg1[subkey])
        bi_prob2.append(bi_marg2[subkey])

    concat_bi_prob1.extend(bi_prob1)
    concat_bi_prob2.extend(bi_prob2)

concat_bi_prob1 = np.array(concat_bi_prob1)
concat_bi_prob2 = np.array(concat_bi_prob2)

rmse_mar_bi = np.linalg.norm(concat_bi_prob1 - concat_bi_prob2)/np.sqrt(len(concat_bi_prob1))
ybar_mar_bi = concat_bi_prob1.mean()
srmse_mar_bi = rmse_mar_bi/ybar_mar_bi

print("srmse_mar_bi", srmse_mar_bi)

# unique_dims = {}
# total_ans = 0
# for key in dict_dims.keys():
#     # print(key, len(dict_dims[key]))
#     unique_dims[key] = list(set(dict_dims[key]))
#     total_ans = total_ans + len(unique_dims[key])
#     # print(unique_dims[key])
#
# print(total_ans)
#
# count_dict_total = {}
# for key in dict_dims.keys():
#     count_dict_total[key] = {}
#     for item in unique_dims[key]:
#         count_dict_total[key][item] = 0
#
#     total_num = 0
#     for ans in dict_dims[key]:
#         count_dict_total[key][ans] = count_dict_total[key][ans] + 1
#         total_num = total_num + 1
#     print(total_num)
#     print(key, count_dict_total[key])
#
#     for item in unique_dims[key]:
#         count_dict_total[key][item] = count_dict_total[key][item]/total_num
#     print(key, count_dict_total[key])
#
# bi_marginal_count = {}
#
# keys = list(dict_dims.keys())
# for i in range(len(keys)):
#     for j in range(i+1, len(keys)):
#         bi_marginal_count[f'{keys[i]}:{keys[j]}'] = {}
#         for item1 in unique_dims[keys[i]]:
#             for item2 in unique_dims[keys[j]]:
#                 bi_marginal_count[f'{keys[i]}:{keys[j]}'][f'{item1}:{item2}'] = 0
#
#         total_num = 0
#         for k in range(len(dict_dims[keys[i]])):
#             bi_marginal_count[f'{keys[i]}:{keys[j]}'][f'{dict_dims[keys[i]][k]}:{dict_dims[keys[j]][k]}'] = (
#                     bi_marginal_count[f'{keys[i]}:{keys[j]}'][f'{dict_dims[keys[i]][k]}:{dict_dims[keys[j]][k]}'] + 1)
#             total_num = total_num + 1
#         print(total_num)
#         print(f'{keys[i]}:{keys[j]}', bi_marginal_count[f'{keys[i]}:{keys[j]}'])
#
#         for item1 in unique_dims[keys[i]]:
#             for item2 in unique_dims[keys[j]]:
#                 bi_marginal_count[f'{keys[i]}:{keys[j]}'][f'{item1}:{item2}'] = (
#                         bi_marginal_count[f'{keys[i]}:{keys[j]}'][f'{item1}:{item2}']/total_num)
#
#         print(f'{keys[i]}:{keys[j]}', bi_marginal_count[f'{keys[i]}:{keys[j]}'])
