import torch
from model.model import ENSS


if __name__ == "__main__":
    model = ENSS(scale_factor=2)
    color = torch.randn(1, 3, 256, 256)
    depth = torch.randn(1, 1, 256, 256)
    jitter = torch.randn(1, 2, 256, 256)
    prev_features = torch.randn(1, 1, 256, 256)
    prev_color = torch.randn(1, 3, 256, 256)
    features, color = model(color, depth, jitter, prev_features, prev_color)
    print(features.shape)
    print(color.shape)