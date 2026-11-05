from copula_tools import *
import json
import pandas as pd
from modules import Net_model, Diffusion
import torch
from tqdm import tqdm


def find_nearest(series, target):
    """返回 series 中离 target 最近的值"""
    return (series - target).abs().argmin()


# f = open('./voca_map.json', 'r', encoding='utf-8')
# data_map = json.load(f)
# f.close()
# print(data_map)

# voc_map = {}
# for key, value in data_map.items():
#     voc_map[value] = key

# f1 = open('./10000_gene_acs_1.json', 'r', encoding='utf-8')
# origi_list = json.load(f1)
# f1.close()
# u_values = origi_list
# print(len(u_values))
# print(u_values[:10])


# f = open('./mapped.json', 'r', encoding='utf')
# training_data = json.load(f)
# f.close()

# data_train = pd.DataFrame(training_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])
# copular = CopulaScaler()
# copular.fit(data_train)
# print(copular.ecdf_map)
# u_dist = copular.u_dist


# attr_list = ['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC']
# data = pd.DataFrame(u_values, columns=attr_list)

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

# all_row_df = pd.DataFrame(all_row, columns=attr_list)
# all_row_inv = copular.inverse_transform(all_row_df)


# all_row_decode = []
# for i in range(len(all_row_inv)):
#     one_row = []
#     for col in all_row_inv.columns:
#         one_value = all_row_inv.iloc[i][col]
#         one_row.append(voc_map[int(one_value)])
#     if i < 10:
#         print(one_row)
#     all_row_decode.append(one_row)

# ff = open('./copula_popu_gene_acs_1_10000.json', 'w', encoding='utf-8')
# json.dump(all_row_decode, ff, ensure_ascii=False)
# ff.close()


if __name__ == '__main__':
    print('begin generating...')
    save_json_file = './100000_gene_acs_17.json'
    model = Net_model(c_in=9, c_out=9, time_dim=512, device="cuda").to(torch.device('cuda'))
    checkpoint = torch.load('./training/DDPM_cond_gene_acs_1/ckpt_1481.pt')
    model.load_state_dict(checkpoint)

    model.eval()
    diffusion = Diffusion(noise_steps=2000, beta_start=1e-4, beta_end=0.02, channels=9, emb_dim=512, device=torch.device('cuda'))

    q = 100000//300
    r = 100000 % 300
    all_sample = []
    for _ in range(q):
        sampled_items = diffusion.sample(model, n=300)
        # sample_out = sampled_items.detach().cpu().numpy()
        sample_out = sampled_items.tolist()
        all_sample.extend(sample_out)

    sampled_items = diffusion.sample(model, n=r)
    # sample_out = sampled_items.detach().cpu().numpy()
    sample_out = sampled_items.tolist()
    all_sample.extend(sample_out)

    f = open(save_json_file, 'w')
    json.dump(all_sample, f)

    f.close()
