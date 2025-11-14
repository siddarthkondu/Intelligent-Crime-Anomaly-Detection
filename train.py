# src/train.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_loaders
from models import get_model
from utils import save_checkpoint

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, _, _ = get_loaders(args.data_root, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers)
    model = get_model(device, latent_dim=args.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    criterion = nn.MSELoss()
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} train loss: {epoch_loss:.6f}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        # optional validation: compute mean reconstruction loss on val set
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs = imgs.to(device)
                    recon = model(imgs)
                    val_loss += criterion(recon, imgs).item() * imgs.size(0)
            val_loss = val_loss / len(val_loader.dataset)
            print(f"Epoch {epoch} val loss: {val_loss:.6f}")
            writer.add_scalar('Loss/val', val_loss, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint({'model_state': model.state_dict(),
                                 'optimizer_state': optimizer.state_dict(),
                                 'epoch': epoch}, os.path.join(args.checkpoint_dir, 'best.pth'))
        else:
            # save every few epochs
            if epoch % args.save_every == 0:
                save_checkpoint({'model_state': model.state_dict(),
                                 'optimizer_state': optimizer.state_dict(),
                                 'epoch': epoch}, os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pth'))

    writer.close()
    # final save
    save_checkpoint({'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict(),
                     'epoch': args.epochs}, os.path.join(args.checkpoint_dir, 'final.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data', help='path to data directory')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--save_every', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
