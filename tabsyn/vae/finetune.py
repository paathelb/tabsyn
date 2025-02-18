import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings

import os
from tqdm import tqdm
import json
import time

from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model
from utils_train import preprocess, TabularDataset

warnings.filterwarnings('ignore')


LR = 1e-3
WD = 0
D_TOKEN = 4
TOKEN_BIAS = True

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z, mu_pretrained, std_pretrained, label=None):   #4096x14, 4096x10, 4096x14, 4096x10, 4096x25x4, 4096x25x4
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0
    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx]) # single value
            x_hat = x_cat.argmax(dim = -1)      # 4096
        acc += (x_hat == X_cat[:,idx]).float().sum()
        total_num += x_hat.shape[0]
    
    ce_loss /= (idx + 1)
    acc /= total_num
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp() 

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())

    # Deviation from Major Distribution Loss            # changed by HP
    mu_pretrained = mu_pretrained.unsqueeze(0).repeat(mu_z.shape[0],1,1)
    std_factor = 3
    
    if label is not None:
        # Why not norm .mean()
        dmd_loss = torch.max(std_factor * std_pretrained.mean() - torch.norm(mu_z-mu_pretrained, p=2, dim=(1,2), keepdim=True).mean(), torch.zeros_like(mu_z)) * label.unsqueeze(1).unsqueeze(2).repeat(1,mu_pretrained.shape[1],1)
        dmd_loss += torch.max(torch.norm(mu_z-mu_pretrained, p=2, dim=(1,2), keepdim=True).mean() - std_factor * std_pretrained.mean(), torch.zeros_like(mu_z)) * (label==0).unsqueeze(1).unsqueeze(2).repeat(1,mu_pretrained.shape[1],1)
    else:
        dmd_loss = torch.max(std_factor * std_pretrained.mean() - torch.norm(mu_z-mu_pretrained, p=2, dim=(1,2), keepdim=True).mean(), torch.zeros_like(mu_z))
    dmd_loss = dmd_loss.mean()

    return mse_loss, ce_loss, loss_kld, acc, dmd_loss


def main(args):
    dataname = args.dataname
    data_dir = f'data/{dataname}'

    max_beta = args.max_beta
    min_beta = args.min_beta
    lambd = args.lambd

    device =  args.device


    info_path = f'data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}' 
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_save_path = f'{ckpt_dir}/model.pt'
    encoder_save_path = f'{ckpt_dir}/encoder.pt'
    decoder_save_path = f'{ckpt_dir}/decoder.pt'

    X_num, X_cat, categories, d_numerical = preprocess(data_dir, task_type = info['task_type'])     # preprocess the tabular data

    X_train_num, _ = X_num
    X_train_cat, _ = X_cat

    X_train_num, X_test_num = X_num     # 27000x14, 3000x14 for 'default'
    X_train_cat, X_test_cat = X_cat     # 27000x10, 3000x10 for 'default'

    # X_train_num = torch.tensor(X_train_num).float()
    # X_train_cat =  torch.tensor(X_train_cat)

    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)

    train_data = TabularDataset(X_train_num.float(), X_train_cat)

    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)
    
    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    # changed by HP
    model = Model_VAE(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR, bias = True, encoder_downsampling=args.encoder_downsampling)
    model = model.to(device)

    # changed by HP
    load_ckpt = True
    model_checkpoint_path = '/home/hpaat/imbalanced_data/tabsyn/tabsyn/vae/ckpt/default/major_only_1_dim_z/model.pt'                    # hard coded
    if load_ckpt and os.path.isfile(model_checkpoint_path):
        print("LOADING MODEL CHECKPOINT")
        model.load_state_dict(torch.load(model_checkpoint_path))
        # changed by HP
        train_z_major = np.load('/home/hpaat/imbalanced_data/tabsyn/tabsyn/vae/ckpt/default/major_only_1_dim_z/train_z_major.npy')      # 21031 x 2 x 4
        train_z_major = torch.from_numpy(train_z_major).cuda()
        mu_pretrained = train_z_major.mean(0)   # 24 x 4
        std_pretrained = train_z_major.std(0)   # 24 x 4

    # changed by HP
    pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR, encoder_downsampling=args.encoder_downsampling).to(device)
    pre_decoder = Decoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)

    num_epochs = 4000           #changed 
    best_train_loss = float('inf')

    current_lr = optimizer.param_groups[0]['lr']
    patience = 0

    beta = max_beta
    start_time = time.time()
    ignore_target = False      # changed by HP
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0
        curr_loss_dmd = 0.0

        curr_count = 0

        for batch_num, batch_cat in pbar:           # 4096x14, 4096x10
            model.train()
            optimizer.zero_grad()

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)

            # changed by HP
            if ignore_target:
                Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat[:,1:]) # 4096x14 #len 10 # 4096x25x4 # 4096x25x4
                # TODO How do we know if batch_cat is 0 or 1
                loss_mse, loss_ce, loss_kld, train_acc, loss_dmd = compute_loss(batch_num, batch_cat[:,1:], Recon_X_num, Recon_X_cat, mu_z, std_z, mu_pretrained, std_pretrained, batch_cat[:,0])  # single values
            else:
                Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat) # 4096x14 #len 10 # 4096x25x4 # 4096x25x4
                loss_mse, loss_ce, loss_kld, train_acc, loss_dmd = compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z, mu_pretrained, std_pretrained)  # single values

            loss = loss_mse + loss_ce + beta * loss_kld + loss_dmd
            loss.backward()
            optimizer.step()

            batch_length = batch_num.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_loss_kl    += loss_kld.item() * batch_length
            curr_loss_dmd    += loss_dmd.item() * batch_length

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        kl_loss = curr_loss_kl / curr_count
        dmd_loss = curr_loss_dmd / curr_count
        
        # num_loss = np.around(curr_loss_gauss / curr_count, 5)
        # cat_loss = np.around(curr_loss_multi / curr_count, 5)
        # kl_loss = np.around(curr_loss_kl / curr_count, 5)
        
        train_loss = num_loss + cat_loss
        scheduler.step(train_loss)

        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != current_lr:
            current_lr = new_lr
            print(f"Learning rate updated: {current_lr}")
            
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            patience = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience += 1
            if patience == 10:
                if beta > min_beta:
                    beta = beta * lambd

        '''
            Evaluation 
        '''
        model.eval()
        with torch.no_grad():
            # changed by HP
            if ignore_target:
                Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num, X_test_cat[:,1:]) # 4096x14 #len 9 # 4096x25x4 # 4096x25x4
                val_mse_loss, val_ce_loss, val_kl_loss, val_acc, val_dmd_loss = compute_loss(X_test_num, X_test_cat[:,1:], Recon_X_num, Recon_X_cat, mu_z, std_z, mu_pretrained, std_pretrained, X_test_cat[:,0])
            else:
                Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num, X_test_cat)
                val_mse_loss, val_ce_loss, val_kl_loss, val_acc, val_dmd_loss = compute_loss(X_test_num, X_test_cat, Recon_X_num, Recon_X_cat, mu_z, std_z, mu_pretrained, std_pretrained)
            val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()

            scheduler.step(val_loss)

        # print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Train ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, train_acc.item()))
        print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Train DMD:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Val DMD:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, dmd_loss, val_mse_loss.item(), val_ce_loss.item(), val_dmd_loss.item(), train_acc.item(), val_acc.item() ))

    end_time = time.time()
    print('Training time: {:.4f} mins'.format((end_time - start_time)/60))
    
    # Saving latent embeddings
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        X_train_num = X_train_num.to(device)
        X_train_cat = X_train_cat.to(device)

        print('Successfully load and save the model!')

        if ignore_target:
            train_z = pre_encoder(X_train_num, X_train_cat[:,1:]).detach().cpu().numpy()
        else:
            train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()

        np.save(f'{ckpt_dir}/train_z.npy', train_z)

        print('Successfully save pretrained embeddings in disk!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Variational Autoencoder')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--max_beta', type=float, default=1e-2, help='Initial Beta.')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum Beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Decay of Beta.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'