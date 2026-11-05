

# import matplotlib.pyplot as plt
# import matplotlib
# import numpy as np

# # 使用非交互式后端避免Qt冲突
# matplotlib.use('Agg')

# # 设置Times New Roman字体
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['axes.unicode_minus'] = False

# # 创建图形和坐标轴，使用更精确的布局控制
# fig = plt.figure(figsize=(14, 6))
# # 手动设置坐标轴位置：[left, bottom, width, height]
# ax = fig.add_axes([0.04, 0.7, 0.92, 0.86])  # 给顶部留出标题空间
# ax.axis('tight')
# ax.axis('off')

# # 表格数据
# table_data = [
#     ["Level", "Name", "Description", "Number of Values"],
#     ["Individual", "AGEP", "Age of person", "Age range from 0 to 99 years"],
#     ["", "SEX", "Gender of person", "2 (Male, Female)"],
#     ["", "RAC1P", "Race of person", "9 (Various races and combinations)"],
#     ["", "ESR", "Employment status", "7 (Different employment statuses, including Nan)"],
#     ["Household", "HINCP", "Household income over the past year", "6 (Income ranges)"],
#     ["", "HHT", "Household/family type", "8 (Types of household arrangements, including Nan)"],
#     ["", "NP", "Number of persons in the household", "7 (Ranges of persons in a household)"],
#     ["", "WIF", "Workers in family during the past 12 months", "5 (Number of workers, including Nan)"],
#     ["", "HUPAC", "Household presence and age of children", "5 (Household with/without children and their ages, includeing Nan)"]
# ]

# # 创建表格，使用bbox精确控制位置
# table = ax.table(cellText=table_data, 
#                 loc='center', 
#                 cellLoc='left',
#                 colWidths=[0.1, 0.15, 0.3, 0.35],
#                 bbox=[0, 0.05, 1, 0.9])  # [x0, y0, width, height]，y0从0.05开始给标题留空间

# # 设置表格样式
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1, 1.5)

# # 设置所有单元格背景为白色
# for i in range(len(table_data)):
#     for j in range(4):
#         table[(i, j)].set_facecolor('white')

# # 设置标题行样式
# for i in range(4):
#     table[(0, i)].set_facecolor('white')
#     table[(0, i)].get_text().set_color('black')
#     table[(0, i)].get_text().set_fontweight('bold')
#     table[(0, i)].get_text().set_fontsize(11)

# # 隐藏所有列分隔线
# for i in range(len(table_data)):
#     for j in range(4):
#         if j < 4:  # 只隐藏前三列之间的分隔线
#             table[(i, j)].visible_edges = 'BT'

# # 定义行分组
# individual_rows = [1, 2, 3, 4]
# household_rows = [5, 6, 7, 8, 9]

# # 设置合并单元格的文本属性
# table[(1, 0)].get_text().set_fontweight('bold')
# table[(5, 0)].get_text().set_fontweight('bold')

# # 隐藏组内行分隔线
# for j in range(0, 4):
#     table[(individual_rows[0], j)].visible_edges = 'T'
        
# for row_idx in individual_rows[1:]:
#     if row_idx > individual_rows[0]:
#         table[(row_idx, 0)].visible_edges = ''
#         for j in range(1, 4):
#             table[(row_idx, j)].visible_edges = ''

# for row_idx in household_rows[1:]:
#     if row_idx > household_rows[0]:
#         table[(row_idx, 0)].visible_edges = ''
#         for j in range(1, 4):
#             table[(row_idx, j)].visible_edges = ''
            
# for j in range(0, 4):
#     table[(household_rows[-1], j)].visible_edges = 'B'

# # 设置组间分隔线
# for j in range(4):
#     table[(4, j)].visible_edges = 'B'
#     table[(5, j)].visible_edges = 'T'

# # 设置表格边框样式
# for key, cell in table.get_celld().items():
#     cell.set_edgecolor('black')
#     cell.set_linewidth(0.5)

# # 精确控制标题位置，使其紧挨表格
# fig.text(0.5, 0.88, 'Table 1',  # y坐标调整到表格正上方
#          fontsize=14, fontweight='bold', 
#          ha='center', va='bottom',  # 使用bottom对齐
#          fontfamily='Times New Roman')

# fig.text(0.5, 0.85, 'Summary of demographic and socioeconomic variables from the ACS survey data', 
#          fontsize=12, 
#          ha='center', va='bottom',  # 使用bottom对齐
#          fontfamily='Times New Roman')

# # 保存图片，不使用tight布局以避免自动调整
# plt.savefig('acs_survey_table_merged_times_new_roman_new.png', dpi=300, 
#             facecolor='white', edgecolor='none')
# print("表格已保存为 'acs_survey_table_merged_times_new_roman_new.png'")

# plt.close(fig)



import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 使用非交互式后端避免Qt冲突
matplotlib.use('Agg')

# 设置Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# 创建图形和坐标轴 - 减小图形高度让表格更紧凑
fig = plt.figure(figsize=(14, 8))  # 进一步减小高度
# 扩大坐标轴区域，几乎占满整个图形 [left, bottom, width, height]
ax = fig.add_axes([0.04, 0.08, 0.92, 0.86])  # 减小边距，增大坐标轴
ax.axis('tight')
ax.axis('off')

# 表格数据
table_data = [
    ["Level", "Name", "Description", "Number of Values"],
    ["Individual", "AGEP", "Age of person", "Age range from 0 to 99 years"],
    ["", "SEX", "Gender of person", "2 (Male, Female)"],
    ["", "RAC1P", "Race of person", "9 (Various races and combinations)"],
    ["", "ESR", "Employment status", "7 (Different employment statuses, including Nan)"],
    ["Household", "HINCP", "Household income over the past year", "6 (Income ranges)"],
    ["", "HHT", "Household/family type", "8 (Types of household arrangements, including Nan)"],
    ["", "NP", "Number of persons in the household", "7 (Ranges of persons in a household)"],
    ["", "WIF", "Workers in family during the past 12 months", "5 (Number of workers, including Nan)"],
    ["", "HUPAC", "Household presence and age of children", "5 (Household w/wo children and their ages, includeing Nan)"]
]

# 创建表格，让表格几乎占满整个坐标轴
table = ax.table(cellText=table_data, 
                loc='center', 
                cellLoc='left',
                colWidths=[0.1, 0.05, 0.29, 0.44],
                bbox=[0, 0.02, 1, 0.96])  # 表格在坐标轴内几乎占满空间

# 设置表格样式 - 减小字体间距让内容更紧凑
table.auto_set_font_size(False)
table.set_fontsize(17)  # 稍微减小字体大小
table.scale(1, 1.8)   # 增加垂直缩放因子，让行更高更紧凑

# 设置所有单元格背景为白色
for i in range(len(table_data)):
    for j in range(4):
        table[(i, j)].set_facecolor('white')

# 设置标题行样式
for i in range(4):
    table[(0, i)].set_facecolor('white')
    table[(0, i)].get_text().set_color('black')
    table[(0, i)].get_text().set_fontweight('bold')
    table[(0, i)].get_text().set_fontsize(17)

# 隐藏所有列分隔线
for i in range(len(table_data)):
    for j in range(4):
        if j < 4:  # 只隐藏前三列之间的分隔线
            table[(i, j)].visible_edges = 'BT'

# 定义行分组
individual_rows = [1, 2, 3, 4]
household_rows = [5, 6, 7, 8, 9]

# 设置合并单元格的文本属性
table[(1, 0)].get_text().set_fontweight('bold')
table[(5, 0)].get_text().set_fontweight('bold')

# 隐藏组内行分隔线
for j in range(0, 4):
    table[(individual_rows[0], j)].visible_edges = 'T'
        
for row_idx in individual_rows[1:]:
    if row_idx > individual_rows[0]:
        table[(row_idx, 0)].visible_edges = ''
        for j in range(1, 4):
            table[(row_idx, j)].visible_edges = ''

for row_idx in household_rows[1:]:
    if row_idx > household_rows[0]:
        table[(row_idx, 0)].visible_edges = ''
        for j in range(1, 4):
            table[(row_idx, j)].visible_edges = ''
            
for j in range(0, 4):
    table[(household_rows[-1], j)].visible_edges = 'B'

for j in range(0, 4):
    table[(household_rows[0], j)].visible_edges = 'T'


# 设置表格边框样式
for key, cell in table.get_celld().items():
    cell.set_edgecolor('black')
    cell.set_linewidth(0.5)

# 标题紧挨表格第一行，位置更精确
fig.text(0.5, 0.97, 'Table 1', 
         fontsize=17, fontweight='bold',  # 减小标题字体
         ha='center', va='bottom',
         fontfamily='Times New Roman')

fig.text(0.5, 0.94, 'Summary of demographic and socioeconomic variables from the ACS survey data', 
         fontsize=17,  # 减小副标题字体
         ha='center', va='bottom',
         fontfamily='Times New Roman')

# 保存图片，不使用bbox_inches='tight'以避免自动添加边距
plt.savefig('acs_survey_table_merged_times_new_roman.png', dpi=300, 
            facecolor='white', edgecolor='none',
            bbox_inches=None, pad_inches=0.1)  # 减小内边距
print("表格已保存为 'acs_survey_table_merged_times_new_roman.png'")

plt.close(fig)