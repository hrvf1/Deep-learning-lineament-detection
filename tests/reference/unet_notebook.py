"""Architecture extraite du notebook MNT, noms des poids conservés."""
import torch
from torch import nn

class DoubleConv(nn.Module):
    """Deux convolutions 3×3, chacune suivie de BatchNorm et ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """Architecture U-Net utilisée dans les expériences du stage."""

    def __init__(self, in_channels=1, out_channels=1, features=(32, 64, 128, 256)):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder_upconv = nn.ModuleList()
        self.decoder_conv = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        precedent = in_channels
        for feature in features:
            self.encoder.append(DoubleConv(precedent, feature))
            precedent = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for feature in reversed(features):
            self.decoder_upconv.append(
                nn.ConvTranspose2d(feature * 2, feature, 2, stride=2)
            )
            self.decoder_conv.append(DoubleConv(feature * 2, feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        connexions = []
        for encodeur in self.encoder:
            x = encodeur(x)
            connexions.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        for index, connexion in enumerate(reversed(connexions)):
            x = self.decoder_upconv[index](x)
            if x.shape != connexion.shape:
                x = nn.functional.interpolate(x, size=connexion.shape[2:])
            x = torch.cat((connexion, x), dim=1)
            x = self.decoder_conv[index](x)
        return self.final_conv(x)
