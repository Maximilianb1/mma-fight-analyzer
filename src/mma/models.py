"""Model definitions: fight/no-fight gate and the two phase+pressure architectures."""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

from . import config as C

MODEL_INPUT_STATS = {
    "r2plus1d": (C.KINETICS_MEAN, C.KINETICS_STD),
    "lstm": (C.IMAGENET_MEAN, C.IMAGENET_STD),
    "gate": (C.IMAGENET_MEAN, C.IMAGENET_STD),
}


def _inflate_conv(conv, in_channels):
    """Replace a pretrained first conv with an N-channel one; extra channels start at zero
    so the pretrained RGB response is preserved at init."""
    cls = type(conv)
    new = cls(in_channels, conv.out_channels, kernel_size=conv.kernel_size,
              stride=conv.stride, padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :conv.in_channels] = conv.weight
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class DualHead(nn.Module):
    def __init__(self, in_features, with_phase=True, with_pressure=True, dropout=0.4):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.phase = nn.Linear(in_features, C.NUM_PHASE_CLASSES) if with_phase else None
        self.pressure = nn.Linear(in_features, C.NUM_PRESSURE_CLASSES) if with_pressure else None

    def forward(self, x):
        x = self.dropout(x)
        return (self.phase(x) if self.phase is not None else None,
                self.pressure(x) if self.pressure is not None else None)


class R2Plus1DDual(nn.Module):
    """Pretrained R(2+1)D-18 (Kinetics-400) with dual classification heads."""

    def __init__(self, in_channels=4, with_phase=True, with_pressure=True,
                 pretrained=True, dropout=0.4):
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)
        if in_channels != 3:
            self.backbone.stem[0] = _inflate_conv(self.backbone.stem[0], in_channels)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.heads = DualHead(feat, with_phase, with_pressure, dropout)

    def forward(self, x):  # x: (B, C, T, H, W)
        return self.heads(self.backbone(x))


class ResNetLSTMDual(nn.Module):
    """Pretrained ResNet-18 frame encoder + LSTM temporal model with dual heads."""

    def __init__(self, in_channels=4, with_phase=True, with_pressure=True,
                 pretrained=True, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = resnet18(weights=weights)
        if in_channels != 3:
            self.encoder.conv1 = _inflate_conv(self.encoder.conv1, in_channels)
        feat = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.lstm = nn.LSTM(feat, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.heads = DualHead(hidden, with_phase, with_pressure, dropout)

    def forward(self, x):  # x: (B, C, T, H, W)
        b, c, t, h, w = x.shape
        feats = self.encoder(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)).view(b, t, -1)
        out, _ = self.lstm(feats)
        return self.heads(out.mean(dim=1))


class GateNet(nn.Module):
    """Frame-level fight/no-fight classifier (1 logit; positive = excluded/non-fight)."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):  # x: (B, 3, H, W)
        return self.backbone(x).squeeze(1)


def build_phase_model(name, in_channels=4, with_phase=True, with_pressure=True,
                      pretrained=True):
    if name == "r2plus1d":
        return R2Plus1DDual(in_channels, with_phase, with_pressure, pretrained)
    if name == "lstm":
        return ResNetLSTMDual(in_channels, with_phase, with_pressure, pretrained)
    raise ValueError(f"unknown model '{name}' (expected r2plus1d or lstm)")


def backbone_and_head_params(model):
    """Split params: pretrained backbone gets a lower LR than freshly initialized parts."""
    backbone = getattr(model, "backbone", None) or getattr(model, "encoder", None)
    backbone_ids = {id(p) for p in backbone.parameters()} if backbone is not None else set()
    bb = [p for p in model.parameters() if id(p) in backbone_ids]
    head = [p for p in model.parameters() if id(p) not in backbone_ids]
    return bb, head


def save_checkpoint(path, model, meta):
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_phase_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = build_phase_model(meta["model_name"], meta["in_channels"],
                              meta.get("with_phase", True), meta["with_pressure"],
                              pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), meta


def load_gate_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GateNet(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["meta"]
