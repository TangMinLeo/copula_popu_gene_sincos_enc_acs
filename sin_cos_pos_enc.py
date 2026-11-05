# 示例：生成位置编码并恢复中点值
import numpy as np
from scipy.optimize import least_squares
import torch
import json
import pandas as pd
from copula_tools import *
from utilities import BatchPositionEncoder
from torch.utils.data import DataLoader
from utilities import CustomDataset
from scipy.optimize import minimize


# def recover_midpoint_least_squares(pe, omegas, x0=0.0):
#     """
#     使用非线性最小二乘法从位置编码中恢复原始值
#     :param pe: 位置编码向量 (numpy数组，形状 [d_model])
#     :param omegas: 频率列表 [ω_0, ω_1, ..., ω_{d_model//2 - 1}]
#     :param x0: 初始猜测值
#     :return: 恢复的中点值 x
#     """
#
#     # 定义残差函数（修正后）
#     def residuals(x):
#         x_scalar = x[0]  # 提取标量值（x是一维数组）
#         res = []
#         for k, omega in enumerate(omegas):
#             res.append(np.sin(x_scalar * omega) - pe[2 * k])  # 正弦残差（标量）
#             res.append(np.cos(x_scalar * omega) - pe[2 * k + 1])  # 余弦残差（标量）
#         return np.array(res)  # 形状 (d_model,) → 一维数组
#
#     # 调用最小二乘优化器
#     result = least_squares(
#         fun=residuals,
#         x0=np.array([x0]),  # 初始猜测值为一维数组
#         bounds=(0.0, 10.0),
#         method='dogbox'
#     )
#     return result.x[0]  # 返回标量结果
#
#
# d_model = 256
# x_true = 90.51
# omegas = [1.0 / (101 ** (2 * k / d_model)) for k in range(d_model // 2)]
#
# # 生成位置编码（确保pe为一维数组）
# pe = np.zeros(d_model)
# for k, omega in enumerate(omegas):
#     pe[2 * k] = np.sin(x_true * omega)
#     pe[2 * k + 1] = np.cos(x_true * omega)
#
# # 恢复中点值
# x_recovered = recover_midpoint_least_squares(pe, omegas, x0=50.0)
# print(f"真实值: {x_true:.6f}, 恢复值: {x_recovered:.6f}")


# def recover_midpoint(pe, omegas, x_int=50.0, device='cuda', lr=0.01, steps=1000):
#     """
#     从位置编码向量中恢复中点值 x
#     :param pe: 位置编码向量 (d_model,)
#     :param omegas: 频率列表 [ω_0, ω_1, ..., ω_{d_model//2 - 1}]
#     :param lr: 学习率
#     :param steps: 优化步数
#     :return: 恢复的中点值 x
#     """
#     x = torch.nn.Parameter(torch.tensor([x_int], requires_grad=True).to(device))  # 初始猜测值
#
#     optimizer = torch.optim.Adam([x], lr=lr)
#
#     # omegas = [1.0 / ((10000 * dim) ** (2 * k / d_model)) for k in range(d_model // 2)]
#
#     for _ in range(steps):
#         optimizer.zero_grad()
#         loss = torch.tensor(0.0).to(device)
#         for k, omega in enumerate(omegas):
#             sin_pred = torch.sin(x * omega)
#             cos_pred = torch.cos(x * omega)
#             loss = loss + (sin_pred - pe[2 * k]) ** 2 + (cos_pred - pe[2 * k + 1]) ** 2
#         loss.backward()
#         optimizer.step()
#
#     return x.item()
#
#
# # 示例：生成位置编码并恢复中点值
# d_model = 512
# x_true = 9.751  # 假设原始中点值为50
#
#
# dim = 2
# omegas = [1.0 / ((100*dim) ** (2 * k / d_model)) for k in range(d_model // 2)]
#
# # 生成位置编码
# pe = torch.zeros(d_model)
# for k, omega in enumerate(omegas):
#     pe[2 * k] = torch.sin(torch.tensor(x_true * omega))
#     pe[2 * k + 1] = torch.cos(torch.tensor(x_true * omega))
#
# # 恢复中点值
# x_recovered = recover_midpoint(pe, omegas, x_int=5.0)
# print(f"真实值: {x_true}, 恢复值: {x_recovered}")


# file_json = open('./wuhan_gene_100000_samples.json', "r")
#
# json_data = json.load(file_json)
# file_json.close()
#
# data = pd.DataFrame.from_dict(json_data, orient='index', columns=['性别', '年龄（分段）', '学历', '人生阶段', '子女年龄',
#                                                                   '是否有车', '通勤方式', '酒店消费水平',
#                                                                   '餐饮消费水平',
#                                                                   '手机品牌', '手机价格（分段）v2', '消费水平（分段）',
#                                                                   '差旅常客', '旅游距离', '出国游'])
# # X = data.head(100)
# # print(data.columns)
# # print(data.head(10))
#
# copula = CopulaScaler()
# copula.fit(data)
#
# transformed_data = copula.transform(data)
#
# data_tensor = torch.from_numpy(transformed_data.values)
# print(data_tensor.shape)
# print(data_tensor[:10, :])
#
# batch_encoder = BatchPositionEncoder(15, 128, device='cpu')
# emb_tensor = torch.zeros(data_tensor.shape[0], data_tensor.shape[1], 128)
#
# q = data_tensor.shape[0] // 32
# # r = data_tensor.shape[0] % 32
# for b in range(1):
#     output_tensor = batch_encoder(data_tensor[(32 * b):(32 * (b + 1)), :])
#     emb_tensor[(32 * b):(32 * (b + 1)), :, :] = output_tensor
#
# # output_tensor = batch_encoder(data_tensor[(32 * q):, :])
# # emb_tensor[(32 * q):, :, :] = output_tensor
#
# print(emb_tensor.shape)
# print(emb_tensor[0, :, :])

def robust_decode(vector, d_model=512, base=10000, freq_weights=None):
    # 1. 提取所有维度的相位信息
    phases = []
    scales = []
    for i in range(d_model // 2):
        scale = base ** (2 * i / d_model)
        scales.append(scale)
        sin_val = vector[2 * i]
        cos_val = vector[2 * i + 1]
        theta = np.arctan2(sin_val, cos_val)
        phase = theta * scale  # 理想相位位置
        phases.append(phase)

    # 2. 设置频率权重（默认高频权重低）
    if freq_weights is None:
        freq_weights = 1 / np.array(scales)  # 低频赋予更高权重

    # 3. 定义损失函数
    def loss(pos):
        error = 0
        for phase, scale, w in zip(phases, scales, freq_weights):
            delta = (pos - phase) % (2 * np.pi * scale)  # 考虑周期缠绕
            delta = min(delta, 2 * np.pi * scale - delta)  # 取最小相位差
            error += w * (delta ** 2)
        return error

    # 4. 全局优化
    result = minimize(
        loss,
        x0=np.median(phases),  # 初始值取相位中位数
        bounds=[(0, 1000)],
        method='L-BFGS-B'
    )

    return result.x[0]


d_model = 128
x_true = 70.0051
base = 101
omegas = [1.0 / (base ** (2 * k / d_model)) for k in range(d_model // 2)]

# 生成位置编码（确保pe为一维数组）
pe = np.zeros(d_model)
for k, omega in enumerate(omegas):
    pe[2 * k] = np.sin(x_true * omega)
    pe[2 * k + 1] = np.cos(x_true * omega)

# 恢复中点值
x_recovered = robust_decode(pe, d_model=d_model, base=base, freq_weights=None)
print(f"真实值: {x_true:.6f}, 恢复值: {x_recovered:.6f}")


# def robust_decode(vector, d_model=256, base=101, freq_weights=None):
#     eps = 1e-12
#     phases = []
#     scales = []
#
#     # 修复1：频率方向与编码严格一致
#     for i in range(d_model // 2):
#         k = i  # 与编码时的k对应
#         exponent = 2 * k / d_model
#         omega = 1.0 / (base ** exponent)  # 与编码时的omega一致
#         scale = 1 / omega  # 关键修复：scale = base^exponent
#
#         sin_val = np.clip(vector[2 * i], -1 + eps, 1 - eps)
#         cos_val = np.clip(vector[2 * i + 1], -1 + eps, 1 - eps)
#
#         theta = np.arctan2(sin_val, cos_val)
#         phase = theta / omega  # 等价于 theta * scale
#         phases.append(phase)
#         scales.append(scale)
#
#     # 修复2：初始值策略
#     valid_scales = [s for s in scales if s < 1000]  # 选择真正低频维度
#     if valid_scales:
#         x0 = np.percentile(phases[:len(valid_scales)], 75)  # 取前25%低频相位的中上值
#     else:
#         x0 = 500
#
#     # 修复3：动态权重分配
#     if freq_weights is None:
#         freq_weights = np.array([1.0 / (s + eps) for s in scales])  # 低频高权重
#
#     # 修复4：增强损失函数
#     def loss(pos):
#         total = 0.0
#         for phase, scale, w in zip(phases, scales, freq_weights):
#             wrapped_pos = pos % (2 * np.pi * scale)
#             wrapped_phase = phase % (2 * np.pi * scale)
#             delta = min(abs(wrapped_pos - wrapped_phase),
#                         2 * np.pi * scale - abs(wrapped_pos - wrapped_phase))
#             total += w * delta ** 2
#         return total
#
#     # 修复5：使用全局优化器
#     result = minimize(
#         loss,
#         x0=x0,
#         method='trust-constr',
#         bounds=[(0, 2000)],
#         options={'verbose': 0, 'gtol': 1e-8}
#     )
#
#     return result.x[0]
#
#
# # --------------------------
# # 验证测试
# d_model = 256
# x_true = 501.2
# base = 101
#
# # 生成编码（保持与之前一致）
# omegas = [1.0 / (base ** (2 * k / d_model)) for k in range(d_model // 2)]
# pe = np.zeros(d_model)
# for k, omega in enumerate(omegas):
#     pe[2 * k] = np.sin(x_true * omega)
#     pe[2 * k + 1] = np.cos(x_true * omega)
#
# # 解码
# x_recovered = robust_decode(pe, d_model=256, base=101)
# print(f"真实值: {x_true:.3f}, 恢复值: {x_recovered:.3f}")
