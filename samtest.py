import torch
from torch.nn.functional import layer_norm
from model.model import ENSS
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = ENSS(scale_factor=2)
    lr_color = torch.randn(1, 3, 256, 256)
    depth = torch.randn(1, 1, 256, 256)
    jitter = torch.randn(1, 2, 1, 1)
    prev_jitter = torch.randn(1, 2, 1, 1)
    motion = torch.randn(1, 2, 256, 256)
    prev_features = torch.randn(1, 1, 512, 512)
    prev_color = torch.randn(1, 3, 512, 512)
    features, hr_color = model(lr_color, depth, motion, jitter, prev_jitter, prev_features, prev_color)
    print(features.shape)
    print(hr_color.shape)
    plt.imshow(hr_color.permute(2, 3, 1))
    