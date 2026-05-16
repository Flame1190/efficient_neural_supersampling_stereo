import argparse
import torch
from torch.nn.functional import layer_norm
from model.model import ENSS
import matplotlib.pyplot as plt


def load_image_as_tensor(path):
    img = plt.imread(path)
    if img.dtype != 'float32' and img.dtype != 'float64':
        img = img.astype('float32') / 255.0
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_color_png", type=str, default="Images/input.png", help="Path to a PNG image to use as lr_color")
    parser.add_argument("--prev_color_png", type=str, default="Images/input_prev.png", help="Path to a PNG image to use as prev_color")
    args = parser.parse_args()

    torch.manual_seed(0)

    model = ENSS(scale_factor=2)

    if args.lr_color_png:
        lr_color = load_image_as_tensor(args.lr_color_png)
    else:
        lr_color = torch.rand(1, 3, 256, 256)
    depth = torch.rand(1, 1, 256, 256)  # torch.rand(1, 1, 256, 256)
    jitter = torch.rand(1, 2, 1, 1)
    prev_jitter = torch.rand(1, 2, 1, 1)
    motion = torch.zeros(1, 2, 256, 256)# torch.rand(1, 2, 256, 256)
    prev_features = torch.rand(1, 1, 512, 512)
    if args.prev_color_png:
        prev_color = load_image_as_tensor(args.prev_color_png)
    else:
        prev_color = torch.rand(1, 3, 512, 512)
    features, hr_color = model(lr_color, depth, motion, jitter, prev_jitter, prev_features, prev_color)
    print(features.shape)
    print(hr_color.shape)
    hr_color = torch.squeeze(hr_color, 0).permute(1, 2, 0).detach().numpy()


    plt.subplot(2, 2, 1)
    plt.imshow(lr_color.squeeze(0).permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title('Low-Resolution Color Image')
    plt.subplot(2, 2, 2)
    plt.imshow(hr_color)
    plt.axis('off')
    plt.title('High-Resolution Color Image')

    plt.subplot(2, 2, 3)
    plt.imshow(depth.squeeze(0).squeeze(0).numpy(), cmap='gray')
    plt.axis('off')
    plt.title('Depth Map')
    plt.subplot(2, 2, 4)
    padding = torch.zeros(motion.shape[2], motion.shape[3])
    motion_with_padding = torch.cat([motion.squeeze(0), padding.unsqueeze(0)], dim=0)
    plt.imshow(motion_with_padding.permute(1, 2, 0).numpy(), cmap='gray')
    plt.axis('off')
    plt.title('Motion Vectors')

    plt.show()