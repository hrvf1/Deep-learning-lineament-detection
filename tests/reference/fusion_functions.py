"""Définitions extraites du script fusion fourni ; réservées à la comparaison."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

def extraire_patchs_pack(
    pack,
    masque,
    taille,
    stride
):
    """
    pack    : (C, H, W)
    masque  : (H, W)

    Retours
    -------
    X : (N, C, taille, taille)
    Y : (N, taille, taille)
    positions : liste des positions (ligne, colonne)
    """

    X = []
    Y = []
    positions = []

    patches_ignores = 0

    _, hauteur, largeur = pack.shape

    for i in range(
        0,
        hauteur - taille + 1,
        stride
    ):
        for j in range(
            0,
            largeur - taille + 1,
            stride
        ):
            patch_x = pack[
                :,
                i:i + taille,
                j:j + taille
            ]

            patch_y = masque[
                i:i + taille,
                j:j + taille
            ]

            if not np.isfinite(patch_x).all():
                patches_ignores += 1
                continue

            X.append(patch_x)
            Y.append(patch_y)
            positions.append((i, j))


    X = np.asarray(
        X,
        dtype=np.float32
    )

    Y = np.asarray(
        Y,
        dtype=np.float32
    )


    print(
        "Patches ignorés à cause de valeurs non valides :",
        patches_ignores
    )

    if patches_ignores > 0:
        print(
            "ATTENTION : si le nombre de patches diffère de tes "
            "expériences précédentes, le split ne sera pas "
            "strictement identique malgré la même seed."
        )

    return X, Y, positions

def normaliser_multi(
    X,
    stats
):
    X_norm = X.copy()

    for c, (p_low, p_high) in enumerate(stats):
        X_norm[:, c] = np.clip(
            (
                X[:, c] - p_low
            ) / (
                p_high - p_low + 1e-10
            ),
            0,
            1
        )

    return X_norm.astype(
        np.float32
    )

def augmenter_patchs_multi(
    X,
    Y
):
    """
    Même transformation appliquée aux 8 canaux et au masque.
    Total : 6 versions par patch.
    """

    X_aug = []
    Y_aug = []

    for x, y in zip(X, Y):
        # Rotations 0°, 90°, 180° et 270°
        for k in range(4):
            X_aug.append(
                np.rot90(
                    x,
                    k,
                    axes=(1, 2)
                )
            )

            Y_aug.append(
                np.rot90(
                    y,
                    k
                )
            )

        # Flip horizontal
        X_aug.append(
            x[:, :, ::-1]
        )

        Y_aug.append(
            np.fliplr(y)
        )

        # Flip vertical
        X_aug.append(
            x[:, ::-1, :]
        )

        Y_aug.append(
            np.flipud(y)
        )


    return (
        np.asarray(
            X_aug,
            dtype=np.float32
        ),
        np.asarray(
            Y_aug,
            dtype=np.float32
        )
    )

class LineamentDataset(Dataset):

    def __init__(
        self,
        X,
        Y
    ):
        self.X = torch.from_numpy(
            np.ascontiguousarray(X)
        ).float()

        self.Y = torch.from_numpy(
            np.ascontiguousarray(Y)
        ).unsqueeze(1).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.Y[idx]
        )

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):

    def __init__(
        self,
        in_channels=8,
        out_channels=1,
        features=[32, 64, 128, 256]
    ):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder_upconv = nn.ModuleList()
        self.decoder_conv = nn.ModuleList()

        self.pool = nn.MaxPool2d(
            2,
            2
        )

        previous_channels = in_channels

        for feature in features:
            self.encoder.append(
                DoubleConv(
                    previous_channels,
                    feature
                )
            )

            previous_channels = feature

        self.bottleneck = DoubleConv(
            features[-1],
            features[-1] * 2
        )

        for feature in reversed(features):
            self.decoder_upconv.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2
                )
            )

            self.decoder_conv.append(
                DoubleConv(
                    feature * 2,
                    feature
                )
            )

        self.final_conv = nn.Conv2d(
            features[0],
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        skips = []

        for encoder_block in self.encoder:
            x = encoder_block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skips = skips[::-1]

        for idx in range(
            len(self.decoder_upconv)
        ):
            x = self.decoder_upconv[idx](x)

            skip = skips[idx]

            if x.shape[2:] != skip.shape[2:]:
                x = torch.nn.functional.interpolate(
                    x,
                    size=skip.shape[2:]
                )

            x = torch.cat(
                (skip, x),
                dim=1
            )

            x = self.decoder_conv[idx](x)

        return self.final_conv(x)

class DiceLoss(nn.Module):

    def __init__(
        self,
        eps=1e-7
    ):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits,
        target
    ):
        predictions = torch.sigmoid(
            logits
        ).reshape(-1)

        target = target.reshape(-1)

        intersection = (
            predictions * target
        ).sum()

        dice = (
            2 * intersection + self.eps
        ) / (
            predictions.sum()
            + target.sum()
            + self.eps
        )

        return 1 - dice

def iou_score(
    logits,
    target,
    seuil=0.5,
    eps=1e-7
):
    """
    IoU calculé sur un batch, comme dans les notebooks précédents.
    """

    prediction = (
        torch.sigmoid(logits) >= seuil
    )

    cible = (
        target >= 0.5
    )

    intersection = (
        prediction & cible
    ).sum().item()

    union = (
        prediction | cible
    ).sum().item()

    return (
        intersection + eps
    ) / (
        union + eps
    )

def recuperer_probabilites(
    model,
    loader,
    device
):
    model.eval()

    probabilites_total = []
    cibles_total = []

    for images, masques in loader:
        images = images.to(
            device,
            dtype=torch.float32
        )

        logits = model(
            images
        )

        probabilites = torch.sigmoid(
            logits
        )

        probabilites_total.append(
            probabilites.cpu().reshape(-1)
        )

        cibles_total.append(
            masques.cpu().reshape(-1)
        )


    probabilites_total = torch.cat(
        probabilites_total
    ).numpy()

    cibles_total = torch.cat(
        cibles_total
    ).numpy()


    return (
        probabilites_total,
        cibles_total
    )

def calculer_metriques(
    probabilites,
    cibles,
    seuil
):
    predictions = (
        probabilites >= seuil
    )

    cibles = (
        cibles >= 0.5
    )


    tp = np.logical_and(
        predictions,
        cibles
    ).sum()

    fp = np.logical_and(
        predictions,
        np.logical_not(cibles)
    ).sum()

    fn = np.logical_and(
        np.logical_not(predictions),
        cibles
    ).sum()

    tn = np.logical_and(
        np.logical_not(predictions),
        np.logical_not(cibles)
    ).sum()


    eps = 1e-8


    iou = tp / (
        tp + fp + fn + eps
    )

    dice = 2 * tp / (
        2 * tp + fp + fn + eps
    )

    precision = tp / (
        tp + fp + eps
    )

    rappel = tp / (
        tp + fn + eps
    )

    specificite = tn / (
        tn + fp + eps
    )

    accuracy = (
        tp + tn
    ) / (
        tp + tn + fp + fn + eps
    )


    return {
        "seuil": float(seuil),
        "IoU": float(iou),
        "Dice_F1": float(dice),
        "Precision": float(precision),
        "Recall": float(rappel),
        "Specificite": float(specificite),
        "Accuracy": float(accuracy),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    }
