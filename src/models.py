"""CIFAR-sized supervised image classification backbones for SafeShift.

All models take [B, 3, 32, 32] images and return raw [B, num_classes]
logits. Normalise inputs in the dataset transforms, not here.

Architecture names refer to structures, NOT downloadable/pretrained checkpoints.
In particular ``resnext29_augmix_arch`` is not automatically checkpoint-
compatible with the RobustBench AugMix ResNeXt implementation: its original
module names and internal input normalisation must be matched separately.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Type

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "gn":
        groups = min(32, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown normalisation {kind!r}; choose 'bn' or 'gn'")


def _conv3(in_ch: int, out_ch: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 3, stride, 1, groups=groups, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, ch: int, stride: int = 1, norm: str = "bn",
                 shortcut_type: str = "projection") -> None:
        super().__init__()
        self.conv1 = _conv3(in_ch, ch, stride)
        self.bn1 = _norm(norm, ch)
        self.conv2 = _conv3(ch, ch)
        self.bn2 = _norm(norm, ch)
        if stride == 1 and in_ch == ch:
            self.shortcut = nn.Identity()
        elif shortcut_type == "projection":
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, ch, 1, stride, bias=False), _norm(norm, ch))
        else:
            raise ValueError("Only shortcut_type='projection' is implemented")

    def forward(self, x: Tensor) -> Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=False)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.shortcut(x), inplace=False)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, ch: int, stride: int = 1, norm: str = "bn") -> None:
        super().__init__()
        out_ch = ch * 4
        self.conv1 = nn.Conv2d(in_ch, ch, 1, bias=False)
        self.bn1 = _norm(norm, ch)
        self.conv2 = _conv3(ch, ch, stride)
        self.bn2 = _norm(norm, ch)
        self.conv3 = nn.Conv2d(ch, out_ch, 1, bias=False)
        self.bn3 = _norm(norm, out_ch)
        self.shortcut = (nn.Identity() if stride == 1 and in_ch == out_ch else
                         nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                                       _norm(norm, out_ch)))

    def forward(self, x: Tensor) -> Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=False)
        y = F.relu(self.bn2(self.conv2(y)), inplace=False)
        return F.relu(self.bn3(self.conv3(y)) + self.shortcut(x), inplace=False)


class _ResNet(nn.Module):
    def __init__(self, block: Type[nn.Module], layers: List[int], channels: List[int],
                 stem_ch: int, num_classes: int, in_channels: int = 3,
                 norm: str = "bn", shortcut_type: str = "projection") -> None:
        super().__init__()
        self.conv1 = _conv3(in_channels, stem_ch)
        self.bn1 = _norm(norm, stem_ch)
        self.relu = nn.ReLU(inplace=False)
        self.inplanes = stem_ch
        for i, (count, width) in enumerate(zip(layers, channels)):
            blocks = []
            for j in range(count):
                kwargs = {"norm": norm}
                if block is _BasicBlock:
                    kwargs["shortcut_type"] = shortcut_type
                blocks.append(block(self.inplanes, width, 2 if i and not j else 1, **kwargs))
                self.inplanes = width * block.expansion
            setattr(self, f"layer{i+1}", nn.Sequential(*blocks))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.inplanes, num_classes)
        self._initialise()

    def _initialise(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        idx = 1
        while hasattr(self, f"layer{idx}"):
            x = getattr(self, f"layer{idx}")(x)
            idx += 1
        return self.fc(torch.flatten(self.avgpool(x), 1))


class CIFARResNet18(_ResNet):
    def __init__(self, num_classes: int = 100, in_channels: int = 3,
                 norm: str = "bn") -> None:
        super().__init__(_BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512],
                         64, num_classes, in_channels, norm)


class CIFARResNet20(_ResNet):
    def __init__(self, num_classes: int = 100, in_channels: int = 3,
                 shortcut_type: str = "projection") -> None:
        super().__init__(_BasicBlock, [3, 3, 3], [16, 32, 64],
                         16, num_classes, in_channels, "bn", shortcut_type)


class CIFARResNet26(_ResNet):
    def __init__(self, num_classes: int = 100, in_channels: int = 3,
                 shortcut_type: str = "projection") -> None:
        super().__init__(_BasicBlock, [4, 4, 4], [16, 32, 64],
                         16, num_classes, in_channels, "bn", shortcut_type)


class CIFARResNet50(_ResNet):
    def __init__(self, num_classes: int = 100, in_channels: int = 3,
                 norm: str = "bn") -> None:
        super().__init__(_Bottleneck, [3, 4, 6, 3], [64, 128, 256, 512],
                         64, num_classes, in_channels, norm)


class CIFARResNet50GN(CIFARResNet50):
    def __init__(self, num_classes: int = 100, in_channels: int = 3) -> None:
        super().__init__(num_classes, in_channels, norm="gn")


class _PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = _conv3(in_ch, out_ch, stride)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = _conv3(out_ch, out_ch)
        self.shortcut = (nn.Identity() if in_ch == out_ch and stride == 1
                         else nn.Conv2d(in_ch, out_ch, 1, stride, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        y = F.relu(self.bn1(x), inplace=False)
        skip = self.shortcut(y) if not isinstance(self.shortcut, nn.Identity) else x
        y = self.conv1(y)
        return self.conv2(F.relu(self.bn2(y), inplace=False)) + skip


class CIFARPreActResNet18(nn.Module):
    def __init__(self, num_classes: int = 100, in_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = _conv3(in_channels, 64)
        in_ch = 64
        for i, out_ch in enumerate((64, 128, 256, 512)):
            blocks = []
            for j in range(2):
                blocks.append(_PreActBasicBlock(in_ch, out_ch, 2 if i and not j else 1))
                in_ch = out_ch
            setattr(self, f"layer{i+1}", nn.Sequential(*blocks))
        self.bn_final = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        for i in range(1, 5):
            x = getattr(self, f"layer{i}")(x)
        x = F.relu(self.bn_final(x), inplace=False)
        return self.fc(torch.flatten(self.avgpool(x), 1))


class _ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, planes: int, stride: int,
                 cardinality: int, base_width: int) -> None:
        super().__init__()
        inner = cardinality * math.floor(planes * base_width / 64)
        if inner <= 0:
            raise ValueError("ResNeXt bottleneck inner width must be positive")
        self.conv1 = nn.Conv2d(in_ch, inner, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inner)
        self.conv2 = _conv3(inner, inner, stride, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(inner)
        self.conv3 = nn.Conv2d(inner, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.shortcut = (nn.Identity() if stride == 1 and in_ch == planes * 4
                         else nn.Sequential(nn.Conv2d(in_ch, planes * 4, 1, stride, bias=False),
                                            nn.BatchNorm2d(planes * 4)))

    def forward(self, x: Tensor) -> Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=False)
        y = F.relu(self.bn2(self.conv2(y)), inplace=False)
        return F.relu(self.bn3(self.conv3(y)) + self.shortcut(x), inplace=False)


class CIFARResNeXt29(nn.Module):
    """29-layer CIFAR ResNeXt; grouped width=(planes*base_width//64)*cardinality.

    This is an architecture-level implementation. RobustBench checkpoint key names,
    preprocessing, and specific reference implementation require separate verification.
    """
    def __init__(self, num_classes: int = 100, cardinality: int = 4,
                 base_width: int = 32, in_channels: int = 3) -> None:
        super().__init__()
        if cardinality < 1 or base_width < 1:
            raise ValueError("cardinality and base_width must be >= 1")
        self.cardinality, self.base_width = cardinality, base_width
        self.conv1 = _conv3(in_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)
        in_ch = 64
        for i, planes in enumerate((64, 128, 256)):
            blocks = []
            for j in range(3):
                blocks.append(_ResNeXtBottleneck(in_ch, planes, 2 if i and not j else 1,
                                                  cardinality, base_width))
                in_ch = planes * 4
            setattr(self, f"layer{i+1}", nn.Sequential(*blocks))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=False)
        for i in range(1, 4):
            x = getattr(self, f"layer{i}")(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))


class CIFARResNeXt29AugMix(CIFARResNeXt29):
    """ResNeXt-29 (4x32 constructor) architecture; NOT an AugMix-trained model."""
    def __init__(self, num_classes: int = 100, in_channels: int = 3) -> None:
        super().__init__(num_classes, cardinality=4, base_width=32,
                         in_channels=in_channels)


class CIFARResNeXt29_8x64d(CIFARResNeXt29):
    def __init__(self, num_classes: int = 100, in_channels: int = 3) -> None:
        super().__init__(num_classes, cardinality=8, base_width=64,
                         in_channels=in_channels)


class _WideBasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, dropout_rate: float) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = _conv3(in_ch, out_ch, stride)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = _conv3(out_ch, out_ch)
        self.dropout_rate = dropout_rate
        self.shortcut = (nn.Identity() if in_ch == out_ch and stride == 1
                         else nn.Conv2d(in_ch, out_ch, 1, stride, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.bn1(x), inplace=False)
        skip = x if isinstance(self.shortcut, nn.Identity) else self.shortcut(x)
        h = self.conv1(h)
        h = F.relu(self.bn2(h), inplace=False)
        if self.dropout_rate:
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return self.conv2(h) + skip


class CIFARWideResNet(nn.Module):
    def __init__(self, depth: int = 28, widen_factor: int = 10,
                 num_classes: int = 100, dropout_rate: float = 0.0,
                 in_channels: int = 3) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0 or depth < 10:
            raise ValueError("WideResNet depth must satisfy depth=6n+4, n>=1")
        if widen_factor < 1 or not 0 <= dropout_rate < 1:
            raise ValueError("Invalid widen_factor or dropout_rate")
        n = (depth - 4) // 6
        self.conv1 = _conv3(in_channels, 16)
        in_ch = 16
        for i, ch in enumerate((16*widen_factor, 32*widen_factor, 64*widen_factor)):
            blocks = []
            for j in range(n):
                blocks.append(_WideBasicBlock(in_ch, ch, 2 if i and not j else 1,
                                              dropout_rate))
                in_ch = ch
            setattr(self, f"layer{i+1}", nn.Sequential(*blocks))
        self.bn = nn.BatchNorm2d(in_ch)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        for i in range(1, 4):
            x = getattr(self, f"layer{i}")(x)
        x = F.relu(self.bn(x), inplace=False)
        return self.fc(torch.flatten(self.avgpool(x), 1))


class CIFARWideResNet28_10(CIFARWideResNet):
    def __init__(self, num_classes: int = 100, dropout_rate: float = 0.0,
                 in_channels: int = 3) -> None:
        super().__init__(28, 10, num_classes, dropout_rate, in_channels)


class CIFARWideResNet28_4(CIFARWideResNet):
    def __init__(self, num_classes: int = 100, dropout_rate: float = 0.0,
                 in_channels: int = 3) -> None:
        super().__init__(28, 4, num_classes, dropout_rate, in_channels)


class CIFARWideResNet40_2(CIFARWideResNet):
    def __init__(self, num_classes: int = 100, dropout_rate: float = 0.0,
                 in_channels: int = 3) -> None:
        super().__init__(40, 2, num_classes, dropout_rate, in_channels)


class _DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth_rate: int = 12, bottleneck: int = 4,
                 dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, bottleneck * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck * growth_rate)
        self.conv2 = _conv3(bottleneck * growth_rate, growth_rate)
        self.dropout_rate = dropout_rate

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.relu(self.bn1(x), inplace=False))
        h = self.conv2(F.relu(self.bn2(h), inplace=False))
        if self.dropout_rate:
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return torch.cat((x, h), dim=1)


class _DenseTransition(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(nn.BatchNorm2d(in_ch), nn.ReLU(inplace=False),
                         nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.AvgPool2d(2))


class CIFARDenseNetBC100_12(nn.Module):
    def __init__(self, num_classes: int = 100, in_channels: int = 3,
                 dropout_rate: float = 0.0, compression: float = 0.5) -> None:
        super().__init__()
        if not 0 < compression <= 1:
            raise ValueError("compression must be in (0, 1]")
        self.conv1 = _conv3(in_channels, 24)
        n_ch = 24
        for group_index in range(3):
            block = []
            for _ in range(16):
                block.append(_DenseLayer(n_ch, 12, 4, dropout_rate))
                n_ch += 12
            setattr(self, f"denseblock{group_index+1}", nn.Sequential(*block))
            if group_index < 2:
                reduced = math.floor(n_ch * compression)
                setattr(self, f"transition{group_index+1}",
                        _DenseTransition(n_ch, reduced))
                n_ch = reduced
        self.bn = nn.BatchNorm2d(n_ch)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_ch, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        for i in range(1, 4):
            x = getattr(self, f"denseblock{i}")(x)
            if i < 3:
                x = getattr(self, f"transition{i}")(x)
        x = F.relu(self.bn(x), inplace=False)
        return self.fc(torch.flatten(self.avgpool(x), 1))


class CIFARVGG16BN(nn.Module):
    def __init__(self, num_classes: int = 100, in_channels: int = 3) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        ch = in_channels
        for repeats, out_ch in zip((2, 2, 3, 3, 3), (64, 128, 256, 512, 512)):
            for _ in range(repeats):
                layers.extend((_conv3(ch, out_ch), nn.BatchNorm2d(out_ch),
                               nn.ReLU(inplace=False)))
                ch = out_ch
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(torch.flatten(self.avgpool(self.features(x)), 1))


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "resnet18_cifar": CIFARResNet18,
    "resnet20_cifar": CIFARResNet20,
    "resnet26_cifar": CIFARResNet26,
    "resnet50_cifar": CIFARResNet50,
    "resnet50_gn_cifar": CIFARResNet50GN,
    "preact_resnet18_cifar": CIFARPreActResNet18,
    "resnext29_augmix_arch": CIFARResNeXt29AugMix,
    "resnext29_8x64d": CIFARResNeXt29_8x64d,
    "wrn28_10": CIFARWideResNet28_10,
    "wrn28_4": CIFARWideResNet28_4,
    "wrn40_2": CIFARWideResNet40_2,
    "densenet_bc100_12": CIFARDenseNetBC100_12,
    "vgg16_bn_cifar": CIFARVGG16BN,
}


def create_model(name: str, num_classes: int = 100, **kwargs: object) -> nn.Module:
    """Return an untrained backbone; never downloads pretrained weights."""
    try:
        constructor = MODEL_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; available: {', '.join(list_models())}") from exc
    return constructor(num_classes=num_classes, **kwargs)


def list_models() -> List[str]:
    return sorted(MODEL_REGISTRY)


__all__ = ["CIFARResNet18", "CIFARResNet20", "CIFARResNet26", "CIFARResNet50",
           "CIFARResNet50GN", "CIFARPreActResNet18", "CIFARResNeXt29",
           "CIFARResNeXt29AugMix", "CIFARResNeXt29_8x64d", "CIFARWideResNet",
           "CIFARWideResNet28_10", "CIFARWideResNet28_4", "CIFARWideResNet40_2",
           "CIFARDenseNetBC100_12", "CIFARVGG16BN", "MODEL_REGISTRY", "create_model",
           "list_models"]
