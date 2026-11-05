import torch
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize


def recover_midpoint_least_squares(pe, omegas, x0=0.0):
    """
    使用非线性最小二乘法从位置编码中恢复原始值
    :param pe: 位置编码向量 (numpy数组，形状 [d_model])
    :param omegas: 频率列表 [ω_0, ω_1, ..., ω_{d_model//2 - 1}]
    :param x0: 初始猜测值
    :return: 恢复的中点值 x
    """

    # 定义残差函数（修正后）
    def residuals(x):
        x_scalar = x[0]  # 提取标量值（x是一维数组）
        res = []
        for k, omega in enumerate(omegas):
            res.append(np.sin(x_scalar * omega) - pe[2 * k])  # 正弦残差（标量）
            res.append(np.cos(x_scalar * omega) - pe[2 * k + 1])  # 余弦残差（标量）
        return np.array(res)  # 形状 (d_model,) → 一维数组

    # 调用最小二乘优化器
    result = least_squares(
        fun=residuals,
        x0=np.array([x0]),  # 初始猜测值为一维数组
        bounds=(0.0, 200.0),
        method='dogbox'
    )
    return result.x[0]  # 返回标量结果


# # 示例：生成位置编码并恢复中点值
# d_model = 128
# x_true = 101.1
# omegas = [1.0 / (100 ** (2 * k / d_model)) for k in range(d_model // 2)]
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
        bounds=[(0.0, 100.0)],
        method='L-BFGS-B'
    )

    return result.x[0]


def improved_robust_decode(vector, d_model=512, base=10000, freq_weights=None, value_range=(0, 100)):
    """
    改进的鲁棒解码，专门针对[0,1]放大100倍的编码
    """
    # 1. 提取相位和频率信息（修正可能的计算错误）
    phases = []
    frequencies = []  # 实际频率 = 1/scale
    
    for i in range(d_model // 2):
        # 修正：这里应该是频率，不是尺度
        # freq = base ** (-2 * i / d_model)  # 类似Transformer的频率衰减
        freq = np.exp(2*i * -(np.log(base) / d_model))
        
        frequencies.append(freq)
        
        sin_val = vector[2 * i]
        cos_val = vector[2 * i + 1]
        
        # 计算相位（弧度）
        theta = np.arctan2(sin_val, cos_val)
        
        # 修正：相位到位置的转换应该是 theta / freq
        # 因为编码是 sin(freq * position)
        estimated_position = theta / freq
        phases.append(estimated_position)
    
    # 2. 基于信噪比的智能权重
    if freq_weights is None:
        # 结合频率权重和信号强度权重
        signal_strengths = []
        for i in range(d_model // 2):
            strength = vector[2*i]**2 + vector[2*i+1]**2
            signal_strengths.append(strength)
        
        # 低频抗噪，但也要考虑信号强度
        freq_weights = (1 / np.array(frequencies)) * np.array(signal_strengths)
        freq_weights = freq_weights / np.sum(freq_weights)
    
    # 3. 多分辨率损失函数
    def multi_scale_loss(position):
        total_error = 0
        
        for i, (freq, weight) in enumerate(zip(frequencies, freq_weights)):
            # 预测的sin/cos值
            pred_sin = np.sin(freq * position)
            pred_cos = np.cos(freq * position)
            
            # 实际值
            actual_sin = vector[2*i]
            actual_cos = vector[2*i+1]
            
            # 使用向量距离作为误差（更鲁棒）
            error = (pred_sin - actual_sin)**2 + (pred_cos - actual_cos)**2
            total_error += weight * error
        
        return total_error
    
    # 4. 多起点优化（处理周期模糊）
    best_position = 0
    best_loss = float('inf')
    
    # 智能起点：使用不同频率子集
    start_positions = []
    
    # 低频起点（抗噪）
    low_freq_indices = [i for i, freq in enumerate(frequencies) if freq < np.median(frequencies)]
    if low_freq_indices:
        low_freq_phases = [phases[i] for i in low_freq_indices]
        start_positions.append(np.median(low_freq_phases))
    
    # 添加边界点和随机点
    start_positions.extend([value_range[0], value_range[1], 
                           np.mean(value_range), np.median(phases)])
    start_positions.extend(np.random.uniform(value_range[0], value_range[1], 3))
    
    for start_pos in start_positions:
        try:
            result = minimize(
                multi_scale_loss,
                x0=start_pos,
                bounds=[value_range],
                method='L-BFGS-B'
            )
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_position = result.x[0]
        except:
            continue
    
    return best_position


# d_model = 512
# x_ = 0.001
# x_true = x_ + 0.001*np.random.normal()
# base = 10001
# omegas = [1.0 / (base ** (2 * k / d_model)) for k in range(d_model // 2)]
#
# # 生成位置编码（确保pe为一维数组）
# pe = np.zeros(d_model)
# for k, omega in enumerate(omegas):
#     pe[2 * k] = np.sin(x_true * omega)
#     pe[2 * k + 1] = np.cos(x_true * omega)
#
# # 恢复中点值
# x_recovered = robust_decode(pe, d_model=d_model, base=base, freq_weights=None)
# print(f"真实值: {x_:.6f}, 恢复值: {x_recovered:.6f}")


# def recover_midpoint(pe, omegas, device='cuda', lr=0.01, steps=200):
#     """
#     从位置编码向量中恢复中点值 x
#     :param pe: 位置编码向量 (d_model,)
#     :param omegas: 频率列表 [ω_0, ω_1, ..., ω_{d_model//2 - 1}]
#     :param lr: 学习率
#     :param steps: 优化步数
#     :return: 恢复的中点值 x
#     """
#     x = torch.tensor([0.5], requires_grad=True).to(device)  # 初始猜测值
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


# 示例：生成位置编码并恢复中点值
# d_model = 512
# x_true = 0.3  # 假设原始中点值为50
#
#
# dim = 2
# omegas = [1.0 / ((10000*dim) ** (2 * k / d_model)) for k in range(d_model // 2)]
#
# # 生成位置编码
# pe = torch.zeros(d_model)
# for k, omega in enumerate(omegas):
#     pe[2 * k] = torch.sin(torch.tensor(x_true * omega))
#     pe[2 * k + 1] = torch.cos(torch.tensor(x_true * omega))
#
# # 恢复中点值
# x_recovered = recover_midpoint(pe, omegas)
# print(f"真实值: {x_true}, 恢复值: {x_recovered}")
