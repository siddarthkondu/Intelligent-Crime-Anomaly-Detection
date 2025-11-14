# src/datasets.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SurveillanceImageDataset(Dataset):
    """
    Simple dataset for loading surveillance frames (images).
    Use for train/val/test. Expects directory with images.
    """
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to folder containing images (flat).
        transform: torchvision transforms.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        for fname in os.listdir(root_dir):
            if fname.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
                self.images.append(os.path.join(root_dir, fname))
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, path  # return path useful for evaluation/visualization

def get_loaders(data_root, batch_size=32, img_size=128, num_workers=4):
    """
    data_root structure:
      data_root/train/   -> normal images (train)
      data_root/val/     -> normal images (val) optional
      data_root/test/normal/ -> normal test images
      data_root/test/anomaly/ -> anomalous test images
    """
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    test_normal_dir = os.path.join(data_root, 'test', 'normal')
    test_anom_dir = os.path.join(data_root, 'test', 'anomaly')

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(SurveillanceImageDataset(train_dir, transform),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = None
    if os.path.isdir(val_dir):
        val_loader = DataLoader(SurveillanceImageDataset(val_dir, transform),
                                batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_normal_loader = DataLoader(SurveillanceImageDataset(test_normal_dir, transform),
                                    batch_size=1, shuffle=False, num_workers=num_workers)
    test_anom_loader = DataLoader(SurveillanceImageDataset(test_anom_dir, transform),
                                  batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_normal_loader, test_anom_loader
