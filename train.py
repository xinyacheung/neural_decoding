import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from pathlib import Path

from model import build_decoder
from torch.utils.data import DataLoader
from dataset import SpikeDataset
import warnings
from config import get_config, latest_weights_file_path, get_weights_file_path

import pytorch_ssim

import wandb

def load_dataloader(config):
    
    train_ds = SpikeDataset(config["neuron_ranges"], config["stimuli_list"], categ='train', seed = config["seed"] )
    test_ds = SpikeDataset(config["neuron_ranges"], config["stimuli_list"], categ='test', seed = config["seed"] )
    train_dataloader = DataLoader(train_ds, config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_ds, config["batch_size"], shuffle=True)
    return train_dataloader, test_dataloader

def ssim(img1, img2, device):
    l = []
    for _ in range(5):
        l.append(torch.tensor(1).to(device))
    for _ in range(10):
        l.append(torch.tensor(2).to(device))
    for _ in range(30):
        l.append(torch.tensor(5).to(device))
    all_loss=0
    for i in range(45):
        img_1=img1[:,:,i,:,:]
        img_2=img2[:,:,i,:,:]
        ssim_loss = pytorch_ssim.SSIM(window_size=4, device=device).to(device)
        all_loss+=ssim_loss(img_1, img_2, device)*l[i].to(device)
    return (sum(l)-all_loss)/sum(l)

def run_test(model, test_dataloader, loss_fn1, loss_fn2, global_step, device):
    
    model.eval()
    loss = 0 
    with torch.no_grad():
        for batch in test_dataloader:
            batch_input = batch["input"].to(device)
            batch_output = batch["output"].to(device)
            model_output = model.decode(batch_input)
            loss += loss_fn1(model_output, batch_output, device)+loss_fn2(model_output, batch_output).item()
    print(f'********* Test Loss: {loss} **********')
    
    # log the loss
    wandb.log({'test/loss': loss, 'global_step': global_step})
    
    return loss

def train_model(config):
    
    device = torch.device(config["device"] if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    Path(config['loss_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, test_dataloader = load_dataloader(config)
    model = build_decoder(neuron_dim=config["num_neurons"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.85, last_epoch=-1)
    loss_fn1 = ssim
    loss_fn2 = nn.MSELoss()
    initial_epoch = 0
    global_step = 0
    
    if config['pre_train']:
        model_filename = latest_weights_file_path(config) if config['pre_train']=='latest' else get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state[['model_state_dict']])
        initial_epoch = state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        global_step = state['global_step']
        del state
     
    # define our custom x axis metric
    wandb.define_metric("global_step")
    # define which metrics will be plotted against it
    wandb.define_metric("test/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    
    train_loss = []
    test_loss = []
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')

        temp_loss = 0

        for batch in batch_iterator:
            
            batch_input = batch["input"].to(device) #torch.Size([64, 45, 140])
            batch_output = batch["output"].to(device) #torch.Size([64, 3, 45, 32, 32])
            model_output = model.decode(batch_input) # [bs, C3, D45, H32, W32]

            loss = loss_fn1(model_output, batch_output, device)+loss_fn2(model_output, batch_output)
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            # log the loss
            wandb.log({'train/loss': loss.item(), 'global_step': global_step})
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1  
            temp_loss = temp_loss+ loss.item()
        
        scheduler.step()
        train_loss.append(temp_loss)
        test_loss.append(run_test(model, test_dataloader, loss_fn1, loss_fn2, global_step, device))
        
        # Save the model
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(), # Save the weights of model
            'scheduler_state_dict': scheduler.state_dict()
        }, model_filename)
    
    np.save(str(Path('.') / config['loss_folder'] / config['model_basename'] / 'train_loss.npy'), train_loss)
    np.save(str(Path('.') / config['loss_folder'] / config['model_basename'] / 'test_loss.npy'), test_loss)
    
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    
    wandb.init(
        project="..",
        config=config
    )
    
    train_model(config)
    
    
    
    
