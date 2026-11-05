import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities import PositionalEncoding, BatchPositionEncoder
import math
import logging
from tqdm import tqdm
from inverse_pe import recover_midpoint_least_squares, robust_decode, improved_robust_decode
import numpy as np


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=512):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        # print(f"x is on cuda: {x.is_cuda}")
        # print(f"t is on cuda: {t.is_cuda}")
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=512):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    # d_model表示离散值嵌入为向量时向量的长度
    # size表示一条样本对应着多少个嵌入向量
    def __init__(self, d_model, size):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.size = size
        self.mha = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.ln = nn.LayerNorm([d_model])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([d_model]),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        # x = x.view(-1, self.channels, self.size[0] * self.size[1]).swapaxes(1, 2)
        assert x.size(1) == self.size, '输入的张量中词的个数与模型中的不一致'
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        # return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size[0], self.size[1])
        return attention_value


class ScalarEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(ScalarEmbedding, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # 输入是标量，维度为1
        self.fc2 = nn.Linear(64, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)
        # self.p_relu_1 = nn.PReLU(init=0.01)
        # self.p_relu_2 = nn.PReLU(init=0.01)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CustomizedEmbLayer(nn.Module):
    def __init__(self, in_features, emb_dim, device):
        super(CustomizedEmbLayer, self).__init__()
        self.in_features = in_features
        self.emb_dim = emb_dim
        self.device = device

        # self.stacked_liner_layers = []

        for i in range(in_features):
            sub_liner = ScalarEmbedding(emb_dim)
            layer_name = f'stacked_emb_layers{i}'
            setattr(self, layer_name, sub_liner)
            # sub_liner = nn.Linear(1, emb_dim)
            # self.stacked_liner_layers.append(sub_liner.to(self.device))

    def forward(self, x):
        assert x.size(1) == self.in_features, 'data dimension mismatch'
        assert len(x.shape) == 2, 'input x must be matrix'

        stacked_data = []
        for i in range(self.in_features):
            layer_name = f'stacked_emb_layers{i}'
            sub_out = getattr(self, layer_name)(x[:, i].unsqueeze(1))

            # sub_out = self.stacked_liner_layers[i](x[:, i].unsqueeze(1))
            stacked_data.append(sub_out)

        return torch.stack(stacked_data, dim=1)


class Net_model(nn.Module):
    def __init__(self, c_in=15, c_out=15, time_dim=512, device=torch.device("cuda")):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.in_liner = nn.Linear(time_dim, time_dim)
        # self.in_liner = CustomizedEmbLayer(c_in, time_dim, device)
        # self.in_liner = BatchPositionEncoder(c_in, time_dim, device=device)

        # a reshape operation is in need to transform the original B x time_dim*c_in tensor to a new tensor of shape B x c_in x time_dim
        self.pos_enc = PositionalEncoding(time_dim, 0.0)
        # self.pos_enc = PositionalEncoding(voca_size, 0.0)

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(256, 128)

        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(128, 256)

        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(64, 256)
        # self.sa3 = SelfAttention(256, [9, 12])

        self.bot1 = DoubleConv(256, 512)
        self.botsa1 = SelfAttention(64, 512)
        self.bot2 = DoubleConv(512, 512)
        self.botsa2 = SelfAttention(64, 512)
        self.bot3 = DoubleConv(512, 256)
        self.botsa3 = SelfAttention(64, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 128)
        # self.sa4 = SelfAttention(128, [18, 24])
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(256, 64)
        # self.sa5 = SelfAttention(64, [36, 48])
        self.up3 = Up(128, 64)

        self.sa6 = SelfAttention(512, 64)
        # self.sa6 = SelfAttention(64, [72, 96])

        self.outc = nn.Conv1d(64, c_out, kernel_size=1)
        # here a reshape operation is in need to transform the B x c_out x 512 tensor to a tensor of shape B x c_out*512
        # self.out_liner = nn.Linear(c_out*time_dim, c_out)
        # self.relu = nn.GELU()
        self.last_liner = nn.Linear(512, 512)

    def pos_encoding(self, t, channels):
        # pos_enc = torch.zeros(len(t), channels)
        # for pos in t:
        #     for i in range(channels//2):
        #         pos_enc[pos, 2*i] = math.sin(pos/(10000**((2*i)/channels)))
        #         pos_enc[pos, 2*i+1] = math.cos(pos/(10000**((2*i)/channels)))
        # return pos_enc

        # inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        inv_freq = torch.exp(torch.arange(0, channels, 2, device=self.device) * (-torch.log(torch.tensor(10000)) / channels))
        # print(f'inv_freq  is on cuda: {inv_freq.is_cuda}')
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)

        # pos_enc = torch.zeros(pos_enc_a.size(0), 2*pos_enc_a.size(1)).to(self.device)
        pos_enc = torch.zeros(pos_enc_a.size(0), 2*pos_enc_a.size(1), device=self.device)

        pos_enc[:, 0::2] = pos_enc_a
        pos_enc[:, 1::2] = pos_enc_b
        # pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)

        # div_terms = torch.exp(torch.arange(0, channels, 2, device=self.device) * (-torch.log(torch.tensor(10000.0)) / channels))
        # # print(div_terms.shape)
        # # print(t.shape)
        #
        # angles = t[:, None] * div_terms[None, :]
        # # print(angles.shape)
        #
        # sin_part = torch.sin(angles)
        # cos_part = torch.cos(angles)
        #
        # pos_enc = torch.zeros(t.shape[0], channels, device=self.device)
        # # pos_enc = torch.zeros(t.shape[0], channels).to(self.device)
        # pos_enc[:, 0::2] = sin_part
        # pos_enc[:, 1::2] = cos_part

        return pos_enc

    def forward(self, x, t):
        # print(f't in model forward is on cuda: {t.is_cuda}')
        t = t.unsqueeze(-1).type(torch.float)
        # t = t.type(torch.float)
        # 对时刻的编码0:T-1共T个时刻
        t = self.pos_encoding(t, self.time_dim)
        t.to(self.device)

        # 线性变换之后再加位置编码
        # print(x.dtype)
        # print(self.in_liner.weight.dtype)
        x = x.to(torch.float32)
        # print('x.shape: ', x.shape)
        x0 = self.in_liner(x)
        # x0 = x0.reshape(x.size(0), x.size(1), -1)

        x0 = self.pos_enc(x0)

        # 线性变换之前（one-hot编码基础上直接加）加位置编码
        # x0 = self.pos_enc(x)
        # x0 = self.in_liner(x0)

        x1 = self.inc(x0)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.botsa1(x4)
        x4 = self.bot2(x4)
        x4 = self.botsa2(x4)
        x4 = self.bot3(x4)
        x4 = self.botsa3(x4)

        x_ = self.up1(x4, x3, t)
        x_ = self.sa4(x_)
        x_ = self.up2(x_, x2, t)
        x_ = self.sa5(x_)
        x_ = self.up3(x_, x1, t)
        x_ = self.sa6(x_)
        output = self.outc(x_)

        # output = torch.reshape(output, (output.shape[0], -1))
        # dis_out = self.out_liner(output)
        # dis_out = self.relu(output)
        dis_out = self.last_liner(output)

        return dis_out


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, channels=15, emb_dim=512, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.channels = channels
        self.device = device
        self.emb_dim = emb_dim
        # self.emb_model = Embedding(vocab, emb_size)

        # self.beta = self.prepare_noise_schedule().to(device)
        # self.beta = self.sqrt_beta_schedule().to(device)
        self.beta = self.cosine_beta_schedule().to(device)

        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        print(self.alpha_hat[0])
        print(self.alpha_hat[-1])
        print(self.alpha[-1])
        print(self.beta[-1])

    def cosine_beta_schedule(self, s=0.008):
        steps = self.noise_steps + 1
        x = torch.linspace(0, self.noise_steps, steps)
        alphas_cumprod = torch.cos(((x / self.noise_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas
        # return torch.clip(betas, 0, 0.999)

    def sqrt_beta_schedule(self, s=0.0001):
        x = torch.linspace(0, self.noise_steps-1, self.noise_steps)  # 0,1,...,999
        alphas_cumprod = 1-torch.sqrt(x/self.noise_steps + s)
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas

    # 重写betas使得最后一步全是噪声
    def prepare_noise_schedule(self):
        betas = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        alphas = 1 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        print(alphas_bar_sqrt_T)
        print(alphas_bar_sqrt_0)

        alphas_bar_sqrt = alphas_bar_sqrt-alphas_bar_sqrt_T

        alphas_bar_sqrt = alphas_bar_sqrt * alphas_bar_sqrt_0/(alphas_bar_sqrt_0-alphas_bar_sqrt_T)

        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:]/alphas_bar[:-1]

        alphas = torch.cat([alphas_bar[0:1], alphas])

        betas = 1 - alphas
        return betas

        # return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    # 这里t是一个批量时间，针对一个batch样本，每个样本对应一个噪声时刻t
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        # Ɛ = torch.randn_like(x)
        # return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
        yup = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * yup, yup

    def sample_timesteps(self, n):
        return torch.randint(low=0, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new populations....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.channels, self.emb_dim)).to(self.device)
            for i in tqdm(reversed(range(0, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                predicted_v = model(x, t)
                # predicted_noise = model(x, t)

                # print('predicted_noise: ', predicted_noise)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]

                pre_x_0 = torch.sqrt(alpha_hat)*x - torch.sqrt(1-alpha_hat)*predicted_v

                if i > 0:
                    alpha_hat_minus = self.alpha_hat[t-1][:, None, None]
                    # sigma = (1-alpha_hat_minus)/(1-alpha_hat)*beta
                    noise = torch.randn_like(x)

                    pre_u_0 = (torch.sqrt(alpha_hat_minus)*beta/(1-alpha_hat))*pre_x_0 + (torch.sqrt(alpha)*(1-alpha_hat_minus)/(1-alpha_hat))*x
                    x = pre_u_0 + torch.sqrt(beta) * noise

                    # x = pre_u_0 + torch.sqrt(beta) * noise
                else:
                    x = pre_x_0
                    # sigma = torch.zeros_like(x)
                    # noise = torch.zeros_like(x)

                # x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        # print('generated position coded x: ', x)
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        # softmax = nn.Softmax(dim=-1)

        # out = torch.argmax(x, dim=-1)

        # out = torch.clamp(x, 0.0, 1.0)
        bases = [101, 907, 199, 503, 601, 307, 401, 809, 701]
        # bases = [101, 1301, 601, 503, 1201, 1103, 907, 199, 809, 307, 1009, 401, 701]
        # bases = [2, 29, 13, 11, 31, 37, 17, 3, 19, 5, 41, 7, 23]
        # bases = [1409, 503, 601, 701, 809, 1511, 907, 199, 307, 401, 101, 1009, 1301, 1103, 1201]
        # freq_bases = torch.tensor([101, 199, 307, 401, 503, 601, 701, 809, 907, 1009, 1103, 1201, 1301, 1409, 1511], requires_grad=False)
        # omegas = torch.exp(torch.arange(0, x.shape[2], 2)[:, None] * (-torch.log(freq_bases)[None, :]/x.shape[2])).permute(1, 0)
        # omegas = torch.exp(torch.arange(0, x.shape[2], 2)[:, None] * (-torch.log(10000.0 * torch.arange(1, x.shape[1]+1))[None, :]/x.shape[2])).permute(1, 0)
        # omegas = omegas.to(self.device)

        # omegas_cloned = omegas.clone().detach().cpu().numpy()
        x_cloned = x.clone()

        # print('tensor_max: ', torch.max(x_cloned))
        # print('tensor_min: ', torch.min(x_cloned))

        # x_cloned = torch.clamp(x_cloned, -1.0, 1.0)
        x_cloned = x_cloned.detach().cpu().numpy()
        # print(x_cloned)

        out = np.zeros((x.shape[0], x.shape[1]))
        out_robust = np.zeros((x.shape[0], x.shape[1]))
        
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                out[i, j] = robust_decode(x_cloned[i, j, :], d_model=512, base=bases[j])
                out_robust[i, j] = improved_robust_decode(x_cloned[i, j, :], d_model=512, base=bases[j])

        model.train()

        # return (out/100.0+1.0)/2.0
        return out/100.0, out_robust/100.0
