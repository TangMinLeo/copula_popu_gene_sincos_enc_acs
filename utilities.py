import os
import torch
# import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
# from investigation_function import investigate_functon
import numpy as np
import torch.nn as nn
import math


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=500):
        # 参数max_len表示一个句子的最大长度，也就是最多包含5000个词，d_model表示词向量的维度
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # print(x.shape)
        # print(self.pe.shape)

        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x
        # return self.dropout(x)


def multi_dim_pe(value, dim_idx, d_model=64, device="cuda"):
    freq_base = torch.tensor(10000.0 * (dim_idx + 1)).to(device)  # 维度差异化
    pe = torch.zeros(d_model)
    one_vec = torch.arange(0, d_model, 2).to(device)/d_model * (-torch.log(freq_base)).to(device)
    pe[0::2] = torch.sin(value*torch.exp(one_vec))
    pe[1::2] = torch.cos(value*torch.exp(one_vec))

    # for i in range(d_model):
    #     ratio = torch.exp(-torch.log(freq_base)*(i/d_model))
    #     # ratio = 1 / (freq_base ** (i / d_model))
    #     if i % 2 == 0:
    #         pe[i] = torch.sin(value * ratio)
    #     else:
    #         pe[i] = torch.cos(value * ratio)
    return pe


class BatchPositionEncoder(nn.Module):
    def __init__(self, num_dims, d_model=512, device='cuda'):
        super().__init__()
        self.num_dims = num_dims
        self.d_model = d_model
        self.device = device

        # 预计算各维度的频率基数（无梯度）
        with torch.no_grad():
            # dim_indices = torch.arange(1, num_dims + 1, device=device)  # (num_dims,)
            # freq_bases = 10000.0 * dim_indices  # (num_dims,)
            # freq_bases = torch.tensor([101, 199, 307, 401, 503, 601, 701, 809, 907, 1009, 1103, 1201, 1301, 1409, 1511], device=device)
            # freq_bases = torch.tensor([1409, 503, 601, 701, 809, 1511, 907, 199, 307, 401, 101, 1009, 1301, 1103, 1201], device=device)
            # freq_bases = torch.tensor([101, 1301, 601, 503, 1201, 1103, 907, 199, 809, 307, 1009, 401, 701],
            #                           device=device)
            freq_bases = torch.tensor([101, 907, 199, 503, 601, 307, 401, 809, 701], device=device)
            # freq_bases = torch.tensor([2, 203, 100, 11, 207, 211, 103, 3, 107, 5, 217, 7, 109], device=device)
            # freq_bases = torch.tensor([2, 29, 13, 11, 31, 37, 17, 3, 19, 5, 41, 7, 23], device=device)

            # temp = math.log(freq_bases)

            # 生成div_term矩阵: (num_dims, d_model//2)
            self.div_terms = torch.exp(
                torch.arange(0, d_model, 2, device=device)[:, None] *
                (-torch.log(freq_bases)[None, :] / d_model)).permute(1, 0)  # -> (num_dims, d_model//2)

    def forward(self, m_batch):
        """
        输入:
            m_batch : 中点位置, 形状为 (batch_size, num_dims)
        输出:
            pe_batch: 位置编码, 形状为 (batch_size, num_dims, d_model)
        """

        batch_size = m_batch.size(0)

        with torch.no_grad():  # 静态特征无需梯度
            # 扩展维度用于广播: (batch_size, num_dims, d_model//2)

            m_expanded = m_batch[..., None]  # (B, D, 1)
            div_terms = self.div_terms[None, ...]  # (1, D, d_model//2)

            # 批量计算正弦和余弦部分
            angles = m_expanded * div_terms  # (B, D, d_model//2)
            sin_part = torch.sin(angles)  # (B, D, d_model//2)
            cos_part = torch.cos(angles)  # (B, D, d_model//2)

            # 交错合并正弦和余弦
            pe = torch.zeros(batch_size, self.num_dims, self.d_model,
                             device=self.device)
            pe[..., 0::2] = sin_part
            pe[..., 1::2] = cos_part

        return pe.detach()  # 确保不参与梯度计算


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


# def get_data(args):
#     # transforms = torchvision.transforms.Compose([
#     #     torchvision.transforms.Resize([72, 96]),  # args.image_size + 1/4 *args.image_size
#     #     torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
#     #     torchvision.transforms.ToTensor(),
#     #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     # ])
#     # dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
#
#     out_samples, vocabulary = investigate_functon(args.file_path)
#     # x = torch.LongTensor(out_samples)
#     # out_emb = emb(x)
#     # dataset = CustomDataset(out_emb)
#
#     cate2base = torch.zeros(len(out_samples), len(out_samples[0]), args.voc_size)
#     for b in range(cate2base.size(0)):
#         for i in range(cate2base.size(1)):
#             idx = out_samples[b][i]
#             cate2base[b, i, idx] = 1.0
#             # small = 0.1/(args.voc_size-1)
#             # for j in range(cate2base.size(2)):
#             #     if j == idx:
#             #         cate2base[b, i, j] = 0.9
#             #     else:
#             #         cate2base[b, i, j] = small
#
#     cate2base = cate2base.cuda(torch.device("cuda"))
#
#     dataset = CustomDataset(cate2base)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
#     return dataloader, vocabulary
