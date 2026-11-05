from copula_tools import *
import json
import pandas as pd
from modules import Net_model, Diffusion
import torch
from tqdm import tqdm
import itertools
from importance_resampling import importance_resamp


def find_nearest(series, target):
    """返回 series 中离 target 最近的值"""
    return (series - target).abs().argmin()

def srmse(data1, data2):
    """ Compute Standardized Root Mean Squared Error between two datasets.

    Reference: https://www.researchgate.net/publication/282815687_A_Bayesian_network_approach_for_population_synthesis
    """
    columns = list(data1.columns.values)
    # Relative frequency
    data1_f = data1.value_counts(normalize=True, dropna=False)
    data2_f = data2.value_counts(normalize=True, dropna=False)
    # Total numbers of categories
    Mi = [data1_f.index.get_level_values(l).union(data2_f.index.get_level_values(l)).unique().size for l in range(len(columns))]
    M = np.prod(Mi)
    # SRMSE
    SRMSE = ((data1_f.subtract(data2_f, fill_value=0)**2) * M).sum()**(.5)
    return SRMSE


f = open('./voca_map.json', 'r', encoding='utf-8')
data_map = json.load(f)
f.close()
print(data_map)

voc_map = {}
for key, value in data_map.items():
    voc_map[value] = key

f1 = open('./100000_gene_acs_6_state_puma_try.json', 'r')
origi_list = json.load(f1)
f1.close()
u_values = origi_list
print(len(u_values))
print(u_values[:10])


f = open('./mapped_tn.json', 'r')
training_data = json.load(f)
f.close()

data_train = pd.DataFrame(training_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])
copular = CopulaScaler()
copular.fit(data_train)
print(copular.ecdf_map)
u_dist = copular.u_dist


attr_list = ['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC']
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
gene_data_list = []
for i in range(len(all_row_inv)):
    one_row = []
    one_row_list = []
    for col in all_row_inv.columns:
        one_value = all_row_inv.iloc[i][col]
        one_row.append(voc_map[int(one_value)])
        one_row_list.append(int(one_value))
    if i < 10:
        print(one_row)
    all_row_decode.append(one_row)
    gene_data_list.append(one_row_list)

ff = open('./gene_copula_acs_decoded_ddpm6_300.json', 'w', encoding='utf-8')
json.dump(all_row_decode, ff, ensure_ascii=False)
ff.close()

# f = open('./puma_mapped_scaramento_6707.json', 'r')
# target_data = json.load(f)
# f.close()

# target_pd = pd.DataFrame(target_data, columns=attr_list)

# resampled = importance_resamp(data_train, target_pd, gene_data_list[:len(target_data)])

# columns = all_row_inv.columns

# srmse_dict = {}
# # df = all_row_inv
# for i in range(1, 6):
#     tuples = list(itertools.combinations(columns, i))  # No repeated elements
#     SRMSE = 0 
#     for tuple in tuples:
#         SRMSE += srmse(target_pd.drop(list(columns.difference(tuple)), axis=1), resampled.drop(list(columns.difference(tuple)), axis=1))
#     SRMSE /= len(tuples)
#     srmse_dict["SRMSE "+str(i)] = SRMSE
# # srmse_dict[model]["Zeros"] = sampling_zeros(source, target, df)
# print(srmse_dict)

# srmse_dict_1 = {}

# for i in range(1, 6):
#     tuples = list(itertools.combinations(columns, i))  # No repeated elements
#     SRMSE = 0 
#     for tuple in tuples:
#         SRMSE += srmse(target_pd.drop(list(columns.difference(tuple)), axis=1), data_train.drop(list(columns.difference(tuple)), axis=1))
#     SRMSE /= len(tuples)
#     srmse_dict_1["SRMSE "+str(i)] = SRMSE
# # srmse_dict[model]["Zeros"] = sampling_zeros(source, target, df)
# print(srmse_dict_1)


# if __name__ == '__main__':
#     print('begin generating...')
#     save_json_file = './100000_gene_acs_6_state_puma_try.json'
#     save_json_file_robust = './100000_gene_acs_6_state_puma_try_robust.json'
#     model = Net_model(c_in=9, c_out=9, time_dim=512, device="cuda").to(torch.device('cuda'))
#     checkpoint = torch.load('./training/DDPM_cond_gene_acs_6/ckpt_240_.pt')
#     model.load_state_dict(checkpoint)

#     model.eval()
#     diffusion = Diffusion(noise_steps=2000, beta_start=1e-4, beta_end=0.02, channels=9, emb_dim=512, device=torch.device('cuda'))

#     q = 300//300
#     r = 300 % 300
#     all_sample = []
#     all_sample_robust = []
#     for _ in range(q):
#         sampled_items, robust_samples = diffusion.sample(model, n=300)
#         # sample_out = sampled_items.detach().cpu().numpy()
#         sample_out = sampled_items.tolist()
#         sample_out_robust = robust_samples.tolist()
        
#         all_sample.extend(sample_out)
#         all_sample_robust.extend(sample_out_robust)

#     if r>0:
#         sampled_items, robust_samples = diffusion.sample(model, n=r)
#         # sample_out = sampled_items.detach().cpu().numpy()
#         sample_out = sampled_items.tolist()
#         sample_out_robust = robust_samples.tolist()
#         all_sample.extend(sample_out)
#         all_sample_robust.extend(sample_out_robust)

#     f = open(save_json_file, 'w')
#     json.dump(all_sample, f)
#     f.close()
    
#     f_robust = open(save_json_file_robust, 'w')
#     json.dump(all_sample_robust, f_robust)
#     f_robust.close()