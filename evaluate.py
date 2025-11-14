# src/evaluate.py
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import torch
import os
from utils import compute_errors_for_loader, plot_histogram
from models import get_model
from dataset import get_loaders
import matplotlib.pyplot as plt

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_normal_loader, test_anom_loader = get_loaders(args.data_root, batch_size=1, img_size=args.img_size, num_workers=4)
    model = get_model(device, latent_dim=args.latent_dim)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    normal_errors, normal_paths = compute_errors_for_loader(model, test_normal_loader, device)
    anom_errors, anom_paths = compute_errors_for_loader(model, test_anom_loader, device)

    y_true = np.concatenate([np.zeros_like(normal_errors), np.ones_like(anom_errors)])
    y_scores = np.concatenate([normal_errors, anom_errors])

    # AUC
    auc_score = roc_auc_score(y_true, y_scores)
    print(f"ROC AUC: {auc_score:.4f}")

    # Histogram
    plot_histogram(normal_errors, anom_errors, save_path=os.path.join(args.output_dir, 'error_hist.png'))

    # Save ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('ROC Curve')
    plt.savefig(os.path.join(args.output_dir, 'roc.png'))
    plt.close()

    # Determine threshold (example: you can pick threshold at some percentile of normal_errors)
    threshold = np.percentile(normal_errors, args.threshold_percentile)
    print(f"Using threshold (percentile {args.threshold_percentile} of normal): {threshold:.6f}")

    # Basic detection stats
    preds_normal = normal_errors > threshold
    preds_anom = anom_errors > threshold
    tp = preds_anom.sum()
    fn = (~preds_anom).sum()
    fp = preds_normal.sum()
    tn = (~preds_normal).sum()
    print(f"TP={tp} FN={fn} FP={fp} TN={tn}")

    # Save errors table
    import pandas as pd
    df = pd.DataFrame({
        'path': list(normal_paths) + list(anom_paths),
        'error': np.concatenate([normal_errors, anom_errors]),
        'label': list(np.zeros_like(normal_errors)) + list(np.ones_like(anom_errors))
    })
    df.to_csv(os.path.join(args.output_dir, 'errors.csv'), index=False)
    print(f"Saved results to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='eval_outputs')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--threshold_percentile', type=float, default=95.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(args)
