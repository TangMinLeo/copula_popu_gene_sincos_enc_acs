from tqdm import tqdm
from torch import optim
from utilities import *
import torch
from modules import Net_model, Diffusion
import logging
# from torch.utils.tensorboard import SummaryWriter
import math
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# diffusion = Diffusion()
diffusion = Diffusion(noise_steps=1000, beta_start=1e-4, beta_end=0.02, channels=15, voc_size=84, device='cuda')

model = Net_model(c_in=15, c_out=15, time_dim=512, voca_size=84, device="cuda").to(device)

# ckpt = torch.load("../popu_diffusion/models/DDPM_category7/ckpt_1849.pt")
ckpt = torch.load("../popu_diffusion/models/DDPM_category7/ckpt_1849.pt")

model.load_state_dict(ckpt)

# out = diffusion.sample(model, 300)
# print(out)
f = open('../popu_diffusion/models/DDPM_category7/voca_map.json', 'r', encoding='utf-8')

data = json.load(f)
f.close()

num_gene = 200000

k = num_gene // 300
r = num_gene % 300

file_json = open(f"./wuhan_gene_100000_samples_new.json", "w")

# gene_population = []
gene_pop_dict = {}
sheet_row = 1

for ii in range(k):
    # sub_file_json = open(f"./gene_category5_1/sub_genes_{ii}.json", "w")
    # sub_count = 0
    # sub_dict = {}

    # out = diffusion.sample(model, 300)
    out = diffusion.sample(model, 300)

    sample_out = out.detach().cpu().numpy()
    sample_out = sample_out.tolist()
    for item in sample_out:
        # sub_dict[sub_count] = item
        # sub_count = sub_count + 1

        gene_pop_dict[sheet_row] = item
        # gene_population.append(tuple(item))
        # for j in range(len(item)):
        #     # sheet.write(sheet_row, j, item[j])
        #     excel.cell(sheet_row, j + 1).value = item[j]
        sheet_row = sheet_row + 1
    # json.dump(sub_dict, sub_file_json)
    # sub_file_json.close()

# out = diffusion.sample(model, r)
out = diffusion.sample(model, r)

sample_out = out.detach().cpu().numpy()
sample_out = sample_out.tolist()

for item in sample_out:
    # sub_dict[sub_count] = item
    # sub_count = sub_count + 1

    gene_pop_dict[sheet_row] = item
    # gene_population.append(tuple(item))
    # for j in range(len(item)):
    #     # sheet.write(sheet_row, j, item[j])
    #     excel.cell(sheet_row, j + 1).value = item[j]
    sheet_row = sheet_row + 1

json.dump(gene_pop_dict, file_json, ensure_ascii=False)
file_json.close()


# # print(data)
# sample_out = out.detach().cpu().numpy()
# sample_out = sample_out.tolist()
# for i in range(len(sample_out)):
#     tem = []
#     for j in range(len(sample_out[i])):
#         tem.append(data[str(sample_out[i][j])])
#     print(tem)
