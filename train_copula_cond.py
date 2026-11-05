import json

import numpy as np
import pandas as pd
from copula_tools import *
from utilities import CustomDataset, BatchPositionEncoder
from torch.utils.data import DataLoader
import argparse
import os
from modules import Net_model, Diffusion
import torch
from torch import optim
from tqdm import tqdm
from investigation_function import investigate_functon
import logging


def find_nearest(series, target):
    """返回 series 中离 target 最近的值"""
    return (series - target).abs().argmin()


def get_data(file_path):
    file_json = open(file_path, "r")
    json_data = json.load(file_json)
    file_json.close()
    # out_samples, vocabulary = investigate_functon(excel_file_path)

    # f_voc = open('./training/DDPM_cond_gene_nhts_5/voc_map.json', 'w', encoding='utf')
    # json.dump(vocabulary, f_voc, ensure_ascii=False)
    # f_voc.close()

    data = pd.DataFrame(np.array(json_data), columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])
    # X = data.head(100)
    # print(data.columns)
    print(data.head())

    copula = CopulaScaler()
    copula.fit(data)
    print(copula.ecdf_map)

    transformed_data = copula.transform(data)
    print(transformed_data.head(10))

    data_tensor = torch.from_numpy(transformed_data.values)
    # data_tensor = 2*data_tensor - 1.0
    data_tensor = data_tensor + 0.0001 * torch.randn_like(data_tensor)
    data_tensor = torch.clamp(data_tensor, 0.0, 1.0)
    # 放大100倍,变成[0,100]之间的数
    data_tensor = 100.0*data_tensor
    print(data_tensor.shape)
    print(data_tensor[:10, :])

    #  位置编码基底为[101, 199, 307, 401, 503, 601, 701, 809, 907, 1009, 1103, 1201, 1301, 1409, 1511]
    #  结合上面的放大，这样编码后不同属性间以及同一属性不同数值之间的编码向量相关性比较低
    batch_encoder = BatchPositionEncoder(9, 512, device='cpu')
    emb_tensor = torch.zeros(data_tensor.shape[0], data_tensor.shape[1], 512)

    q = data_tensor.shape[0] // 32
    # r = data_tensor.shape[0] % 32
    for b in range(q):
        output_tensor = batch_encoder(data_tensor[(32 * b):(32 * (b + 1)), :])
        emb_tensor[(32 * b):(32 * (b + 1)), :, :] = output_tensor
    output_tensor = batch_encoder(data_tensor[(32 * q):, :])
    emb_tensor[(32 * q):, :, :] = output_tensor

    print(emb_tensor.shape)
    print(emb_tensor[0, :, :])
    print('emb_tensor_max: ', torch.max(emb_tensor))
    print('emb_tensor_min: ', torch.min(emb_tensor))

    data_tensor = emb_tensor.cuda(torch.device("cuda")).to(torch.float32)
    dataset = CustomDataset(data_tensor)
    
    # batch size change here
    data_loader = DataLoader(dataset, batch_size=48, shuffle=True)
    return data_loader, copula


def launch(data_loader, copula):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_cond_gene_acs_6"

    if os.path.exists(f'./training/{args.run_name}'):
        pass
    else:
        os.makedirs(f'./training/{args.run_name}')

    args.in_c = 9
    args.out_c = 9

    args.emb_dim = 512
    # args.emb_dim = 128

    args.epochs = 260
    # args.epochs = 578
    
    args.batch_size = 48
    # args.image_size = [72, 96]
    # args.dataset_path = r"D:\Download\luce_clean_1011\train"
    # args.file_path = r'D:\Download\Kim and Bansal (2023)_TR-C\data_NHTSint_indiv.csv'
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # args.lr = 3e-4
    args.lr = 0.00015852
    
    args.save_file = f'./training/{args.run_name}/training_process_v_pre_260_epochs.json'

    # data_tensor = torch.from_numpy(pd_data.values)
    # print(data_tensor.shape)
    #
    # batch_encoder = BatchPositionEncoder(args.in_c, args.emb_dim, device='cpu')
    # emb_tensor = torch.zeros(data_tensor.shape[0], data_tensor.shape[1], args.emb_dim)
    #
    # q = data_tensor.shape[0]//32
    # # r = data_tensor.shape[0] % 32
    # for b in range(q):
    #     output_tensor = batch_encoder(data_tensor[(32*b):(32*(b+1)), :])
    #     emb_tensor[(32*b):(32*(b+1)), :, :] = output_tensor
    # output_tensor = batch_encoder(data_tensor[(32 * q):, :])
    # emb_tensor[(32 * q):, :, :] = output_tensor
    #
    # data_tensor = data_tensor.cuda(args.device).to(torch.float32)
    # dataset = CustomDataset(data_tensor)
    #
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    train(args, data_loader, copula)


def train(args, dataloader, copula):
    device = args.device
    print(device)
    model = Net_model(c_in=args.in_c, c_out=args.out_c, time_dim=args.emb_dim, device="cuda").to(device)
    checkpoint = torch.load('./training/DDPM_cond_gene_acs_6/ckpt_240.pt')
    model.load_state_dict(checkpoint)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500, eta_min=1e-7)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    

    mse = torch.nn.MSELoss()

    diffusion = Diffusion(noise_steps=2000, beta_start=1e-4, beta_end=0.02, channels=args.in_c, emb_dim=args.emb_dim, device=device)

    l = len(dataloader)
    print('dataloader len: ', l)

    ff = open(f'./voca_map.json', 'r', encoding='utf-8')
    voc_map = json.load(ff)
    ff.close()

    data_map = {}
    for key, value in voc_map.items():
        data_map[str(int(value))] = key

    gene_dict = {}
    ff = open(args.save_file, "w")
    
    logging.basicConfig(
    filename=f'./training/{args.run_name}/training_process_v_pre_260_epochs.log',  # 每次写入都会追加到文件末尾
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
    

    start_epoch = 0
    for epoch in range(start_epoch, args.epochs + start_epoch):
        print(f'starting epoch {epoch}:')
        pbar = tqdm(dataloader)
        loss_list = []
        for i, samples in enumerate(pbar):
            samples = samples.to(device)
            # print('samples.shape: ', samples.shape)
            # samples = samples + 0.002*torch.randn(samples.shape, device=device)
            # samples = torch.clamp(samples, 0.0, 1.0)
            t = diffusion.sample_timesteps(samples.shape[0]).to(device)

            x_t, noise = diffusion.noise_images(samples, t)

            # print(f't after use in adding noise is on cuda: {t.is_cuda}')
            # predicted_noise = model(x_t, t)
            predicted_v = model(x_t, t)
            v_t = torch.sqrt(diffusion.alpha_hat[t][:, None, None])*noise - torch.sqrt(1-diffusion.alpha_hat[t][:, None, None])*samples
            # print(predicted_noise.dtype)
            # 模型用来拟合v_t
            loss = mse(v_t, predicted_v)
            # loss = mse(noise, predicted_noise)

            # print(loss.dtype)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            copy_loss = loss.clone()
            # copy_loss = copy_loss.detach().item()
            pbar.set_postfix(MSE=copy_loss.item())
            # pbar.set_postfix(MSE=loss.item())

            # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            # print("MSE", loss.item())

            loss_list.append(loss.item())

        # sampled_items = diffusion.sample(model, n=6)
        # print(sampled_items)

        # data = pd.DataFrame(sampled_items, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])

        # u_dist = copula.u_dist
        # # mapping1 = copula.reverse_mapping
        # all_row = []
        # for i in range(len(data)):
        #     one_row = []
        #     for col in data.columns:
        #         key = col
        #         series = pd.Series(u_dist[key])
        #         one_value = data.iloc[i][col]
        #         idx = find_nearest(series, one_value)
        #         one_row.append(series[idx])
        #         # one_row.append(data_map[str(int(mapping1[key][int(idx)]))])
        #     # print(one_row)
        #     all_row.append(one_row)

        # total_row = pd.DataFrame(all_row, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])

        # inversed_cdf = copula.inverse_transform(total_row)

        # sample_out = inversed_cdf.values.tolist()
        # # gene_rep = []
        # for j in range(len(sample_out)):
        #     tem = []
        #     for item in sample_out[j]:
        #         tem.append(data_map[str(int(item))])
        #     # gene_rep.append(tem)
        #     print(tem)
        
        
        # gene_dict[epoch] = [gene_rep, np.mean(np.array(loss_list)), loss_list]
        # gene_dict[epoch] = [sample_out, np.mean(np.array(loss_list)), loss_list]
        
        # RAC1P  1:white 2:black 3:american indan 4:alaska native 5:american indian and alaska natibe 6:asian 7:native hawaiian 8: other races 9:tow or more races
        # ESR   1:civilian employed at work 2:civilian employed not at work 3：unemployed 4:armed force at work 5:armed force not at work 6:not in labor force 7:nan
        # HINCP  1:nan 2:negative 3:[0, 28000) 4:[28000, 55000) 5:[55000, 160000) 6:[160000, inf)
        # HHT   1:married copule household 2:male householder, no spouse 3:female householder, no spouse 4:male living alone 5: male not living along 6:female living along 7: female not living along 8:nan
        # NP   1:1 2:2 3:3 4:4 5:5 6:6 7:[7,inf)
        # WIF  0:no workers 1:1 worker 2: 2 workers 3:3 or more workers 4:nan
        # HUPAC  1:children under 6 2:children 6-17 3:children under 6 and 6-17 4:no own children 5: nan
        
        # save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("training", args.run_name, f"ckpt_{epoch}_.pt"))

        scheduler.step()
        print(f'learning rate: {scheduler.get_last_lr()}')
        gene_dict[epoch] = [np.mean(np.array(loss_list)), loss_list, scheduler.get_last_lr()]
        # gene_dict[epoch] = [sample_out, np.mean(np.array(loss_list)), loss_list, scheduler.get_last_lr()]
        ave_loss = float(np.mean(np.array(loss_list)))
        lr = float(scheduler.get_last_lr()[-1])
        logging.info(f"Epoch {epoch}: loss={ave_loss:.8f}, learning_rate={lr:.8f}")

    json.dump(gene_dict, ff, ensure_ascii=False)
    ff.close()


if __name__ == '__main__':
    print('begin training')
    # json_file_path = "./wuhan_gene_100000_samples.json"
    # dataloader = get_data(json_file_path)
    if os.path.exists('./training/DDPM_cond_gene_acs_6'):
        pass
    else:
        os.makedirs('./training/DDPM_cond_gene_acs_6', exist_ok=True)

    file_path = './mapped_tn.json'
    # excel_file_path = './data_NHTSint_indiv.csv'
    dataloader, copula = get_data(file_path)
    launch(dataloader, copula)
