"""Модель для приложения: класс TorchCNN (копия из тетрадки), загрузка весов и подготовка картинки к виду MNIST."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch import nn


class TorchCNN(nn.Module):
    """CNN по словарю гиперпараметров cfg. Тот же класс, что в lab2_cnn_mnist.ipynb (инициализация здесь не нужна)."""

    def __init__(self, cfg: dict):
        super().__init__()
        bn, k = cfg["batch_norm"], cfg["kernel_size"]
        feats, in_ch = [], 1
        for b in range(cfg["n_blocks"]):
            out_ch = cfg["base_filters"] * 2 ** b
            for _ in range(cfg["convs_per_block"]):
                feats.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=1, padding="same", bias=not bn))
                if bn:
                    feats.append(nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1))
                feats.append(nn.ReLU(inplace=True))
                in_ch = out_ch
            feats.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if cfg["conv_dropout"] > 0:
                feats.append(nn.Dropout(cfg["conv_dropout"]))
        if cfg["head_type"] == "flatten":
            feats.append(nn.Flatten())
        else:
            feats += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.features = nn.Sequential(*feats)
        self.features.eval()
        with torch.no_grad():
            n_feat = self.features(torch.zeros(1, 1, 28, 28)).shape[1]
        head, n_in = [], n_feat
        if cfg["dense_units"] > 0:
            head += [nn.Linear(n_in, cfg["dense_units"]), nn.ReLU(inplace=True)]
            n_in = cfg["dense_units"]
        if cfg["dense_dropout"] > 0:
            head.append(nn.Dropout(cfg["dense_dropout"]))
        head.append(nn.Linear(n_in, 10))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        return self.head(self.features(x))


class DigitClassifier:
    """Загружает torch_best.pt и классифицирует картинки с цифрами."""

    def __init__(self, path: str | Path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.cfg, self.mean, self.std = ckpt["cfg"], ckpt["mean"], ckpt["std"]
        self.model = TorchCNN(self.cfg).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def predict(self, x28: np.ndarray) -> np.ndarray:
        """x28: картинка 28x28 uint8 (светлая цифра на тёмном фоне). Возвращает вероятности 10 классов."""
        x = (x28.astype(np.float32) / 255.0 - self.mean) / self.std
        logits = self.model(torch.from_numpy(x)[None, None])
        return torch.softmax(logits, dim=1)[0].numpy()


def to_mnist(img: np.ndarray) -> np.ndarray:
    """Картинка со светлой цифрой на тёмном фоне (uint8, любой размер) -> 28x28 uint8, как в MNIST:
    обрезка по краям цифры, большая сторона 20 px, центр масс в центре кадра."""
    ys, xs = np.nonzero(img > 30)
    if len(xs) == 0:
        return np.zeros((28, 28), np.uint8)
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 20 / max(h, w)
    crop = np.asarray(Image.fromarray(crop).resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS),
                      dtype=np.float32)
    out = np.zeros((28, 28), np.float32)
    h, w = crop.shape
    out[(28 - h) // 2:(28 - h) // 2 + h, (28 - w) // 2:(28 - w) // 2 + w] = crop
    cy, cx = ndimage.center_of_mass(out)
    out = ndimage.shift(out, (13.5 - cy, 13.5 - cx), order=1, mode="constant")
    return np.clip(out, 0, 255).astype(np.uint8)


def image_to_gray(img) -> np.ndarray:
    """Любая картинка (PIL, массив RGB/RGBA/L, dict из gr.Sketchpad) -> светлая цифра на тёмном фоне, uint8."""
    if isinstance(img, dict):
        img = img.get("composite")
    if img is None:
        return np.zeros((28, 28), np.uint8)
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 4:
        # кладу картинку на белый фон: у холста Gradio вокруг белого листа прозрачная рамка, а штрихи чёрные
        a = img[..., 3:4].astype(np.float32) / 255.0
        gray = (img[..., :3].astype(np.float32) * a + 255.0 * (1 - a)).mean(axis=2)
        if gray.max() - gray.min() < 30 and a.min() < 0.98:   # светлая цифра на прозрачном фоне: она видна только в альфа-канале
            return img[..., 3].astype(np.uint8)
        img = gray
    elif img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    img = img.astype(np.float32)
    # если фон светлый (тёмная цифра на белом), инвертирую
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    if border.mean() > 127:
        img = 255 - img
    return img.astype(np.uint8)
