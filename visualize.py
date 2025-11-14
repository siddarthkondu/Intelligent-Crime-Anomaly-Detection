# src/visualize.py
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image

def show_image_tensor(img_tensor, title=None):
    """
    img_tensor: torch tensor shape (C,H,W), values [0,1]
    """
    img = to_pil_image(img_tensor.cpu())
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

def compare_reconstruction(orig, recon, title=None):
    # orig, recon are tensors (C,H,W)
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(to_pil_image(orig.cpu()))
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(to_pil_image(recon.cpu()))
    plt.title('Reconstruction')
    plt.axis('off')
    if title:
        plt.suptitle(title)
    plt.show()
