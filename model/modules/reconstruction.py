import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.modules import KernelPrediction

class Reconstruction(BaseModel):
    """
    reconstruction network for neural network module
    """

    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 f: int, 
                 m: int, 
                 enc_kernel_predictor: KernelPrediction, 
                 dec_kernel_predictor: KernelPrediction):
        super().__init__()
        # self.jitter_prediction

        self.enc_kernel_predictor = enc_kernel_predictor
        self.dec_kernel_predictor = dec_kernel_predictor


        # Not sure if m includes the convs before and after the jitter-conditoned convs
        # For now, it does not
        # 
        #self.net = nn.Sequential(
        #    nn.Conv2d(in_channels, f, 3, 1, 1),
       #     nn.ReLU(),
       #     *zip(*[(nn.Conv2d(f, f, 3, 1, 1), nn.ReLU()) for _ in range(m)]),
       #     nn.Conv2d(f, out_channels, 3, 1, 1),
       #     nn.ReLU()
       # )
        layers = []

        layers.append(nn.Conv2d(in_channels, f, 3, 1, 1))
        layers.append(nn.ReLU())

        for _ in range(m):
            layers.append(nn.Conv2d(f, f, 3, 1, 1))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(f, out_channels, 3, 1, 1))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, 
                color: torch.Tensor,
                depth: torch.Tensor,
                jitter: torch.Tensor,
                prev_features: torch.Tensor,
                prev_color: torch.Tensor
                ):
        B, C, H, W = color.shape
        assert B == 1 # kernel prediction may break if B > 1
        
        # jitter tensor is B, 2, H, W but its the same for each pixel
        enc_kernel = self.enc_kernel_predictor(jitter[:, :, 0, 0])
        enc_kernel = enc_kernel.repeat(10, 10, 1, 1)
        dec_kernel = self.dec_kernel_predictor(jitter[:, :, 0, 0])
        dec_kernel_mask = dec_kernel.repeat(64, 64, 1, 1)
        dec_kernel_color = dec_kernel.repeat(3, 64, 1, 1)

        dec_kernel = dec_kernel.repeat(64, 64, 1, 1)

        #dec_kernel_mask = 
        # x = torch.cat([color, depth, jitter, prev_features, prev_color], dim=1)
        # Removed prev_color as it is not used in the reconstruction network. Needed only for blending it seems


        # Not sure why jitter needs to be encoded as well, but this is according to the diagram.
        jitter = jitter.expand(-1, -1, H, W)
        x = torch.cat([color, depth, jitter, prev_features], dim=1)
        print(x.shape)
        x = F.conv2d(x, enc_kernel, padding=1)
        x = self.net(x)
        
        features = F.conv2d(x, dec_kernel, padding=1)

        mask = F.conv2d(features, dec_kernel_mask, padding=1)
        color_prior_blending = F.conv2d(features, dec_kernel_color, padding=1)
        
        mask = self.sigmoid(features)
        color_prior_blending = self.relu(features)
        return mask, color_prior_blending, features
        

   