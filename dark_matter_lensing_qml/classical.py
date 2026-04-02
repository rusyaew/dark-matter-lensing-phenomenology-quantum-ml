from __future__ import annotations

import torch
from torch import nn


class CompressedMlpClassifier(nn.Module):
    def __init__(
            self,
            *,
            input_dim: int,
            n_lensing_classes: int,
            hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_lensing_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ResidualBlock(nn.Module):
    def __init__(
            self,
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.GELU()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        residual = self.skip(images)
        features = self.block(images)
        return self.activation(features + residual)


class DeepLenseResidualClassifier(nn.Module):
    def __init__(
            self,
            *,
            n_lensing_classes: int,
            embedding_dim: int = 128,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            ResidualBlock(in_channels=32, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=64, stride=2),
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=128, stride=2),
            ResidualBlock(in_channels=128, out_channels=128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
        )
        self.classifier = nn.Linear(embedding_dim, n_lensing_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        return self.encoder(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(images))


class DeepLenseWideResidualClassifier(nn.Module):
    def __init__(
            self,
            *,
            n_lensing_classes: int,
            embedding_dim: int = 192,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            ResidualBlock(in_channels=48, out_channels=48),
            ResidualBlock(in_channels=48, out_channels=96, stride=2),
            ResidualBlock(in_channels=96, out_channels=96),
            ResidualBlock(in_channels=96, out_channels=192, stride=2),
            ResidualBlock(in_channels=192, out_channels=192),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(192, embedding_dim),
            nn.GELU(),
            nn.Dropout(p=0.25),
        )
        self.classifier = nn.Linear(embedding_dim, n_lensing_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        return self.encoder(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(images))
