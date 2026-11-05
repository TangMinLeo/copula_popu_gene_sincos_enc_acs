
import pandas as pd
import dataframe_image as dfi
import json
import itertools
import numpy as np

def create_clean_table():
    """
    创建完全干净的表格，只有首尾横线
    """
    # 创建数据
    # data = [{'Method':'Training data', 'SRMSE 1': 0.42612503767780496, 'SRMSE 2': 0.9775095298330475, 'SRMSE 3': 1.9260608718694752, 'SRMSE 4': 3.7027968761183487, 'SRMSE 5': 7.174169649117056},
    #         {'Method':'Independent', 'SRMSE 1': 0.4258911701158711, 'SRMSE 2': 1.163464685283664, 'SRMSE 3': 2.3769644788742332, 'SRMSE 4': 4.4642497661955804, 'SRMSE 5': 8.230079907859745},
    #         {'Method':'CTGAN Copula', 'SRMSE 1': 0.25358447213348523, 'SRMSE 2': 0.754933890775964, 'SRMSE 3': 1.6560730110272601, 'SRMSE 4': 3.3400478384531307, 'SRMSE 5': 6.561218395434594},
    #         {'Method':'TVAE Copula','SRMSE 1': 0.2911793321526763, 'SRMSE 2': 0.7373998269963844, 'SRMSE 3': 1.5459207783857754, 'SRMSE 4': 3.0862814848904883, 'SRMSE 5': 6.082040841591567},
    #         {'Method':'BN Copula', 'SRMSE 1': 0.014629452301813792, 'SRMSE 2': 0.27509777491152826, 'SRMSE 3': 0.7864058293113487, 'SRMSE 4': 1.8336757203854501, 'SRMSE 5': 3.997479071311176},
    #         {'Method':'Diffusion', 'SRMSE 1': 0.2462681321956143, 'SRMSE 2': 0.5510504584943811, 'SRMSE 3': 1.100240682150269, 'SRMSE 4': 2.224094622193033, 'SRMSE 5': 4.6618587502129305}]
    
    # data = [{'Method':'Independent', 'SRMSE 1': 0.17484999562761486, 'SRMSE 2': 0.6728219807160583, 'SRMSE 3': 1.502037437589554, 'SRMSE 4': 2.9246434579506357, 'SRMSE 5': 5.472563491335323},
    #         {'Method':'CTGAN Copula', 'SRMSE 1': 0.38539120401468585, 'SRMSE 2': 0.8846121945439511, 'SRMSE 3': 1.6913127913462915, 'SRMSE 4': 3.0928442066380852, 'SRMSE 5': 5.616728201312609},
    #         {'Method':'TVAE Copula', 'SRMSE 1': 0.5235640924231313, 'SRMSE 2': 1.1171312699974845, 'SRMSE 3': 2.004829123101045, 'SRMSE 4': 3.481598518959418, 'SRMSE 5': 6.070029077174351},
    #         {'Method':'BN Copula', 'SRMSE 1': 0.0022916826200274206, 'SRMSE 2': 0.13257886817629377, 'SRMSE 3': 0.36620789087211747, 'SRMSE 4': 0.8259283722276953, 'SRMSE 5': 1.7564170973017907},
    #         {'Method':'Diffusion', 'SRMSE 1': 0.13975639631420705, 'SRMSE 2': 0.2828861935715997, 'SRMSE 3': 0.504320736651166, 'SRMSE 4': 0.895609795258952, 'SRMSE 5': 1.6316960291278224}]
    
    # data = [{'Method':'Independent', 'SRMSE 1': 0.17776285756976654, 'SRMSE 2': 0.6790015440887316, 'SRMSE 3': 1.5701364735022831, 'SRMSE 4': 3.209084862902061, 'SRMSE 5': 6.276384390984214},
    #         {'Method':'CTGAN Copula', 'SRMSE 1': 0.5263093712601544, 'SRMSE 2': 1.166000380506041, 'SRMSE 3': 2.18146790189463, 'SRMSE 4': 3.9368111172781766, 'SRMSE 5': 7.100159889755243},
    #         {'Method':'TVAE Copula', 'SRMSE 1': 0.18489034407450208, 'SRMSE 2': 0.5390173462547706, 'SRMSE 3': 1.2106972115366497, 'SRMSE 4': 2.5144706062298954, 'SRMSE 5': 5.047531836718892},
    #         {'Method':'BN Copula', 'SRMSE 1': 0.01928305469838725, 'SRMSE 2': 0.16854901378843815, 'SRMSE 3': 0.4935117899611907, 'SRMSE 4': 1.1704230434220142, 'SRMSE 5': 2.570847552854124},
    #         {'Method':'Diffusion', 'SRMSE 1': 0.22514026232148698, 'SRMSE 2': 0.49236282261579045, 'SRMSE 3': 0.9227644383117444, 'SRMSE 4': 1.6814646698504876, 'SRMSE 5': 3.0914764576280693}]
    
    data = [{'Method':'Independent', 'Sampling zeros': 27766, 'Structural zeros':  718804, 'Precision':  0.12975828723468566, 'Recall': 0.4542328238696234, 'F1 Score': 0.201854008084617},
            {'Method':'CTGAN Copula', 'Sampling zeros': 47175, 'Structural zeros': 579351, 'Precision':  0.2047912494576442, 'Recall': 0.5549300236685213, 'F1 Score':0.299175018335346},
            {'Method':'TVAE Copula', 'Sampling zeros': 56399, 'Structural zeros':  279600, 'Precision': 0.5411046381272101, 'Recall': 0.7208797986530445, 'F1 Score': 0.618187342434388},
            {'Method':'BN Copula', 'Sampling zeros': 66000, 'Structural zeros': 190085, 'Precision': 0.6980782813577299, 'Recall': 0.8345897111950652, 'F1 Score': 0.760254607078351},
            {'Method':'Diffusion', 'Sampling zeros': 50312, 'Structural zeros': 65973, 'Precision': 0.8589224509660629, 'Recall': 0.8077661522544783, 'F1 Score': 0.832559221874076}]
    
    # data = [{'Method':'Independent', 'Sampling zeros': 51695, 'Structural zeros': 1039362, 'Precision': 0.13315211966576546, 'Recall': 0.4341774084683039, 'F1 Score': 0.203802690964048},
    #         {'Method':'CTGAN Copula', 'Sampling zeros': 104001, 'Structural zeros': 749711, 'Precision':  0.2997120472378494, 'Recall': 0.6306516192419563, 'F1 Score': 0.406322591276675},
    #         {'Method':'TVAE Copula', 'Sampling zeros': 78902, 'Structural zeros':  1023557, 'Precision': 0.13254783431143033, 'Recall': 0.5692972323191231, 'F1 Score': 0.215030692131693},
    #         {'Method':'BN Copula', 'Sampling zeros': 118129, 'Structural zeros': 201646, 'Precision': 0.788943628269548, 'Recall': 0.8393491199285217, 'F1 Score': 0.813366196949627},
    #         {'Method':'Diffusion', 'Sampling zeros': 87519, 'Structural zeros': 84738, 'Precision': 0.8612630966710891, 'Recall': 0.805095312526134, 'F1 Score': 0.832232583524003}]
    
    df = pd.DataFrame(data)
    # df = pd.read_csv('./result_table_old.csv')
    # df = df.rename(columns={'Unnamed: 0':'Method'})
    # df = df.drop(['Zeros'], axis=1)
    
    # new_row = pd.Series({'Method':'Diffusion', 'SRMSE 1': 0.13984328010985933, 'SRMSE 2': 0.2831788208063523, 'SRMSE 3': 0.5043475895246513, 'SRMSE 4': 0.894679584668439, 'SRMSE 5': 1.6283985927787363})
    # df = df._append(new_row, ignore_index=True)
    
    print(list(df.columns))
    print(df)
    # df = pd.DataFrame(data)
    
        # 定义加粗函数
    def highlight_top_two_min(series):
        """
        对每列中最小的两个值加粗
        """
        # 获取最小的两个值的索引
        min_indices = series.nsmallest(2).index
        
        # 创建样式数组
        styles = [''] * len(series)
        for idx in min_indices:
            styles[idx] = 'font-weight: bold'
        
        return styles
    
    def highlight_top_two_max(series):
        """对最大的两个值加粗"""
        max_indices = series.nlargest(2).index
        styles = [''] * len(series)
        for idx in max_indices:
            styles[idx] = 'font-weight: bold'
        return styles
    
    # 自定义CSS样式
    custom_css = [
        # 移除所有边框
        {'selector': '', 'props': [('border', 'none')]},
        # 标题
        {'selector': 'caption',
             'props': [('text-align', 'left'),
                       ('font-family', 'Courier'),
                       ('font-size', '10pt'),
                       ('font-weight', 'normal'),
                       ('padding', '10px 0'),
                       ('caption-side', 'top'),
                       ('white-space', 'pre-line')]},
        # 表头 - 只有底部边框
        {'selector': 'thead th', 'props': [
            ('border-top', '2px solid #000000'),
            ('border-bottom', '1px solid #000000'),
            ('border-left', 'none'),
            ('border-right', 'none'),
            ('background-color', '#FFFFFF'),
            ('color', '#000000'),
            ('font-weight', 'normal'),
            ('text-align', 'center'),
            ('padding', '8px 12px')
        ]},
        # 数据行 - 无边框
        {'selector': 'tbody td', 'props': [
            ('border', 'none'),
            ('background-color', '#FFFFFF'),
            ('text-align', 'center'),
            ('padding', '6px 12px')
        ]},
        # 最后一行 - 只有底部边框
        {'selector': 'tbody tr:last-child td', 'props': [
            ('border-bottom', '2px solid #000000')
        ]},
        # 表格容器 - 无边框
        {'selector': 'table', 'props': [
            ('border', 'none'),
            ('border-collapse', 'collapse'),
            ('background-color', '#FFFFFF')
        ]}
    ]
    
    styled_df = df.style\
        .set_table_styles(custom_css)\
        .format({
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1 Score': '{:.4f}'
        })\
        .set_properties(**{
            'background-color': 'white',
            'color': 'black',
            'font-family': 'Courier',
            'font-size': '10pt'
        })\
        .set_caption(f"Tabel 5 <br> Diversity and feasibility of the synthesized population at the state level")\
        .hide(axis='index')
            # .set_caption(f"Tabel 2 <br> SRMSE for the state level transfer experiment")\
            # .set_caption(f"Tabel 4 <br> SRMSE for the county level transfer experiment")\
            # .set_caption(f"Tabel 5 <br> SRMSE for the TN state to TX state transfer experiment")\
                
            # .set_caption(f"Tabel 3 <br> Diversity and feasibility of the synthesized population at the state level")\
    styled_df = styled_df.apply(highlight_top_two_max, subset=['Precision', 'Recall', 'F1 Score', 'Sampling zeros'])
    styled_df = styled_df.apply(highlight_top_two_min, subset=['Structural zeros'])
    # styled_df = styled_df.apply(highlight_top_two_min, subset=['SRMSE 1', 'SRMSE 2', 'SRMSE 3', 'SRMSE 4', 'SRMSE 5'])
      
    # 导出图片
    dfi.export(
        styled_df,
        "srmse_5.png",
        dpi=300
    )
    
    print("干净风格表格已保存为 clean_academic_table.png")

# 运行
create_clean_table()


# def srmse(data1, data2):
#     """ Compute Standardized Root Mean Squared Error between two datasets.

#     Reference: https://www.researchgate.net/publication/282815687_A_Bayesian_network_approach_for_population_synthesis
#     """
#     columns = list(data1.columns.values)
#     # Relative frequency
#     data1_f = data1.value_counts(normalize=True, dropna=False)
#     data2_f = data2.value_counts(normalize=True, dropna=False)
#     # Total numbers of categories
#     Mi = [data1_f.index.get_level_values(l).union(data2_f.index.get_level_values(l)).unique().size for l in range(len(columns))]
#     M = np.prod(Mi)
#     # SRMSE
#     SRMSE = ((data1_f.subtract(data2_f, fill_value=0)**2) * M).sum()**(.5)
#     return SRMSE


# file_name = './resampled_from_gene_copula_2_decoded_1853429.csv'
# print(file_name)
# df = pd.read_csv(file_name, dtype='int')
# print(df.head())
# gene_data = df.values.tolist()
# print(gene_data[:5])
# print(len(gene_data))

# f = open('./mapped_ca.json', 'r')
# target_data = json.load(f)
# f.close()
# print('len(target_data): ', len(target_data))
# print('target_data[:5] ', target_data[:5])
# data_tar = pd.DataFrame(target_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])

# f = open('./mapped.json', 'r', encoding='utf')
# training_data = json.load(f)
# f.close()
# print('len(training_data): ', len(training_data))
# print('training_data[:5] ', training_data[:5])
# data_train = pd.DataFrame(training_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])

# srmse_dict = {}
# columns = data_train.columns

# for i in range(1, 6):
#     tuples = list(itertools.combinations(columns, i))  # No repeated elements
#     SRMSE = 0 
#     for tuple in tuples:
#         SRMSE += srmse(data_tar.drop(list(columns.difference(tuple)), axis=1), df.drop(list(columns.difference(tuple)), axis=1))
#     SRMSE /= len(tuples)
#     srmse_dict["SRMSE "+str(i)] = SRMSE
# # srmse_dict[model]["Zeros"] = sampling_zeros(source, target, df)
# print('target and resampling ', srmse_dict)