# src/models.py
import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 3 x H x W -> 32 x H/2 x W/2
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # -> 64 x H/4 x W/4
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),# -> 128 x H/8 x W/8
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# -> 256 x H/16 x W/16
            nn.ReLU(True),
        )
        # Bottleneck linear mapping
        self.flatten = nn.Flatten()
        # compute feature map size dynamically in forward if needed; assume input 128x128 => H/16=8 => 256*8*8=16384
        self.fc_enc = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 8 * 8)

        # Decoder (mirrors encoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # -> 128 x H/8 x W/8
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # -> 64 x H/4 x W/4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # -> 32 x H/2 x W/2
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # -> 3 x H x W
            nn.Sigmoid(),  # images normalized to [0,1]
        )

    def forward(self, x):
        batch = x.size(0)
        enc = self.encoder(x)
        enc_flat = self.flatten(enc)
        z = self.fc_enc(enc_flat)
        dec_flat = self.fc_dec(z)
        dec = dec_flat.view(batch, 256, 8, 8)
        out = self.decoder(dec)
        return out

def get_model(device, latent_dim=128):
    model = ConvAutoencoder(latent_dim=latent_dim).to(device)
    return model
