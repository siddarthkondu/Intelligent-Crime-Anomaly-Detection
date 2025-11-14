# src/utils.py
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint

def reconstruction_error(x, x_recon):
    # x, x_recon: tensors, shape (C,H,W)
    return torch.mean((x - x_recon) ** 2).item()

def compute_errors_for_loader(model, loader, device):
    model.eval()
    errors = []
    paths = []
    with torch.no_grad():
        for imgs, path in tqdm(loader, desc="Computing errors"):
            imgs = imgs.to(device)
            recon = model(imgs)
            err = torch.mean((imgs - recon) ** 2, dim=[1,2,3]).cpu().numpy()
            errors.extend(err.tolist())
            paths.extend(path)
    return np.array(errors), paths

def plot_histogram(normal_errors, anomaly_errors, save_path=None):
    plt.figure(figsize=(8,5))
    plt.hist(normal_errors, bins=50, alpha=0.6, label='normal')
    plt.hist(anomaly_errors, bins=50, alpha=0.6, label='anomaly')
    plt.legend()
    plt.xlabel('Reconstruction error (MSE)')
    plt.ylabel('Count')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
