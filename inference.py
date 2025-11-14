# src/inference.py
import torch
from torch.utils.data import DataLoader
from dataset import FrameDataset
from models import ConvAutoencoder
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = FrameDataset(args.data_root, split='test', img_size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = ConvAutoencoder(latent_dim=args.latent).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    scores = []
    labels = []
    paths = []
    with torch.no_grad():
        for batch in tqdm(loader):
            if len(batch) == 3:
                x,label, p = batch
            else:
                x,label = batch
                p = [None]*len(label)
            x = x.to(device)
            out = model(x)
            mse = torch.mean((out - x)**2, dim=[1,2,3]).cpu().numpy()
            scores.extend(mse.tolist())
            labels.extend([int(l) for l in label])
            paths.extend(p)

    # compute AUC
    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = None
    print("AUC:", auc)
    # print top anomalous frames
    idx_sorted = np.argsort(scores)[::-1]
    for i in idx_sorted[:10]:
        print(f"score {scores[i]:.6f} label {labels[i]} path {paths[i]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--latent', type=int, default=256)
    args = parser.parse_args()
    evaluate(args)
