from openpyxl import load_workbook
import xlrd
import pandas as pd
# from sklearn.isotonic import IsotonicRegression
import numpy as np
import csv
import random
import json


def investigate_functon(file_name):

    dict_dims = {}
    idx_map = {}

    with open(file_name, 'r', encoding='utf-8') as f:
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

    unique_dims = {}
    total_ans = 0
    for key in dict_dims.keys():
        unique_dims[key] = list(set(dict_dims[key]))
        total_ans = total_ans + len(unique_dims[key])

    vocabulary = {}
    voca_count = 0
    for key in unique_dims.keys():
        for value in unique_dims[key]:
            vocabulary[f'{key}:{value}'] = voca_count
            voca_count = voca_count + 1

    print(vocabulary)
    print(voca_count)

    f = open('./training/DDPM_cond_gene_nhts_5/training_sample_50000.json', 'w')
    table_input = []
    # random.seed(19)
    np.random.seed(19)
    for i in range(count-1):
        # sampling only the 5% data
        # thre = random.uniform(0.0, 1.0)
        thre = np.random.uniform(0.0, 1.0)
        if thre <= 0.05:
            one_sample = []
            for key in dict_dims.keys():
                one_sample.append(vocabulary[f'{key}:{dict_dims[key][i]}'])
            table_input.append(one_sample)

    json.dump(table_input, f, ensure_ascii=False)
    f.close()
    return table_input, vocabulary


# def investigate_functon(file_name):
#
#     file = load_workbook(filename=file_name)
#     # file = xlrd.open_workbook(filename=file_name)
#
#     sheet = file['Sheet1']
#     # break_cols = [12, 13, 16, 19, 26, 27, 28, 29, 30, 31]
#
#     indicate_of_list_value = {}
#     big_dict = {}
#     total_answers = {}
#
#     for j in range(1, sheet.max_column+1):
#         if '、' in sheet.cell(1, j).value:
#             tep_str = sheet.cell(1, j).value.split('、')[1]
#             tep_str = tep_str.split('：')[0]
#         else:
#             tep_str = sheet.cell(1, j).value
#             tep_str = tep_str.split('：')[0]
#
#         print(tep_str)
#         big_dict[tep_str] = []
#         total_answers[tep_str] = []
#         indicate_of_list_value[tep_str] = 0
#
#     # excel里行列索引都是从1开始
#     for i in range(2, sheet.max_row + 1):
#         for j in range(1, sheet.max_column+1):
#
#             value = sheet.cell(i, j).value
#             if '、' in sheet.cell(1, j).value:
#                 tep_str = sheet.cell(1, j).value.split('、')[1]
#             else:
#                 tep_str = sheet.cell(1, j).value
#
#             tep_str = tep_str.split('：')[0]
#
#             if '┋' in value:
#                 val_list = value.split('┋')
#                 total_answers[tep_str].extend(val_list)
#
#                 big_dict[tep_str].append(val_list)
#                 indicate_of_list_value[tep_str] = 1
#
#             else:
#                 total_answers[tep_str].append(value)
#                 big_dict[tep_str].append(value)
#
#     # print(big_dict)
#     # print(total_answers)
#     print(indicate_of_list_value)
#
#     num_answers = 0
#     vocabulary = {}
#     voca_count = 0
#     keys = ['性别', '年龄（分段）', '学历', '人生阶段', '子女年龄', '是否有车', '通勤方式', '酒店消费水平', '餐饮消费水平',
#             '手机品牌', '手机价格（分段）v2', '消费水平（分段）', '差旅常客', '旅游距离', '出国游']
#     for key in keys:
#         total_answers[key] = list(set(total_answers[key]))
#         num_answers = num_answers + len(total_answers[key])
#         # if key == '子女年龄':
#         #     total_answers[key] = ['6~12岁', '2~5岁', '无子女', '13~18岁', '0~2岁', '19岁以上']
#         # if key == '酒店宾馆':
#         #     total_answers[key] = ['旅馆招待所', '公寓式酒店', '经济型酒店', '度假村', '青年旅社', '星级酒店', '其他酒店宾馆']
#
#         for attrib in total_answers[key]:
#             vocabulary[f'{key}:{attrib}'] = voca_count
#             voca_count = voca_count + 1
#
#     # print(total_answers)
#     print(num_answers)
#     print(vocabulary)
#
#     table_input = []
#     # excel里行列索引都是从1开始
#     for i in range(2, sheet.max_row + 1):
#
#         one_sample = []
#         for j in range(1, sheet.max_column+1):
#
#             if '、' in sheet.cell(1, j).value:
#                 # print(sheet.cell(1, j).value)
#                 tep_str = sheet.cell(1, j).value.split('、')[1]
#             else:
#                 tep_str = sheet.cell(1, j).value
#
#             tep_str = tep_str.split('：')[0]
#
#             # voc_key = f'{tep_str}:{value}'
#             if tep_str in keys:
#                 value = sheet.cell(i, j).value
#
#                 if '┋' in value:
#                     val_list = value.split('┋')
#
#                     answer_tuple = ()
#                     for item in val_list:
#                         voc_key = f'{tep_str}:{item}'
#                         answer_tuple = answer_tuple + (vocabulary[voc_key],)
#
#                     one_sample.append(answer_tuple)
#                     # total_answers[tep_str].extend(val_list)
#                     # big_dict[tep_str].append(val_list)
#                     # indicate_of_list_value[tep_str] = 1
#
#                 else:
#                     # total_answers[tep_str].append(value)
#                     # big_dict[tep_str].append(value)
#                     voc_key = f'{tep_str}:{value}'
#                     one_sample.append(vocabulary[voc_key])
#
#                     # if voc_key in ['酒店消费水平:(空)', '酒店消费水平:(跳过)']:
#                     #     break
#                     # else:
#                     #     one_sample.append(voc_key)
#
#         if len(one_sample) == len(keys):
#             table_input.append(one_sample)
#     # print(table_input)
#
#     return table_input, vocabulary


if __name__ == '__main__':
    file_path = r'D:\我的桌面\社区调查\副本254121960_0_社区人口基础调研问卷（湖口小区）_948_895.xlsx'
    table, vocabulary = investigate_functon(file_path)
    print(len(table))
    keys = ['性别', '年龄（分段）', '学历', '人生阶段', '子女年龄', '是否有车', '通勤方式', '酒店消费水平',
            '餐饮消费水平', '手机品牌', '手机价格（分段）v2', '消费水平（分段）', '差旅常客', '旅游距离', '出国游']
    data = pd.DataFrame(table, columns=keys)
    print(data.head(10))
    x = data['年龄（分段）']
    counts = x.value_counts(normalize=True)
    print(counts)
    x = x.sort_values(ascending=True)

    n = len(x)
    p_empirical = np.arange(1, n + 1) / n

    ir = IsotonicRegression(increasing=True)
    p_smooth = ir.fit_transform(x, p_empirical)

    unique_x = np.unique(x)
    cdf_smooth = {xi: np.max(p_smooth[x == xi]) for xi in unique_x}

    print(cdf_smooth)
    for key in cdf_smooth.keys():
        if key == 2:
            print(cdf_smooth[key])
        else:
            print(cdf_smooth[key]-cdf_smooth[key-1])
