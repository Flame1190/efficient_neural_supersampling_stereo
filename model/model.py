import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.modules import DepthToSpace, SpaceToDepth, Blending, KernelPrediction, Reconstruction
from utils import warp, retrieve_elements_from_indices

class ENSS(BaseModel):
    def __init__(self, scale_factor: int, depth_block_size: int = 3, num_conv_layers: int = 4) -> None:
        super().__init__()
        # Warp module
        self.warp = Warping(scale_factor=scale_factor, depth_block_size=depth_block_size)

        self.reconstruction = Reconstruction(in_channels=3 + 1 + 2 + 1 + 3, out_channels=64, f=64, m=4, enc_kernel_predictor=KernelPrediction(7, 1024, 3), dec_kernel_predictor=KernelPrediction(7, 1024, 3))
        # Then concat (done in forward pass)

        # First conv and ReLU layer
        self.predicted_kernel = KernelPrediction(7, 1024, 3)
        self.relu1 = nn.ReLU()

        
        self.sigmoid = nn.Sigmoid()
        self.relu2 = nn.ReLU()
        # Blending module
        self.blending = Blending()
        self.depth_to_space1 = DepthToSpace(block_size=scale_factor)
        self.depth_to_space2 = DepthToSpace(block_size=scale_factor)

    def forward(self, 
                color: torch.Tensor,
                depth: torch.Tensor,
                jitter: torch.Tensor,
                prev_features: torch.Tensor,
                prev_color: torch.Tensor) -> torch.Tensor:
        B, _, H, W = color.shape
        assert B == 1 # kernel prediction may break if B > 1
        
        #predicted_kernel = self.predicted_kernel(jitter[:, :, 0, 0])
        #predicted_kernel = predicted_kernel.repeat(3, 3, 1, 1)
        
        #x = torch.concat([color, depth, jitter, prev_features, prev_color], dim=1)
        #x = F.conv2d(x, predicted_kernel, padding=1)
        #x = self.relu1(x)
       # x = self.conv_layers(x)

       
        mask, color_prior_blending, features = self.reconstruction(color, depth, jitter, prev_features, prev_color)
        # todo: match up dimensions correctly. This might need some clarification on how the model actually works, as this isn't very clear from the paper
        blending = self.blending(previous_frame=prev_color, current_frame=color_prior_blending, blending_mask=mask)
        new_color = self.depth_to_space2(blending)
        features = self.depth_to_space1(features)


        return features, new_color


class Warping(BaseModel):
    def __init__(self, scale_factor: int, depth_block_size: int = 3) -> None:
        assert depth_block_size % 2 == 1 # They used 8x8 but I can't use even kernels yet
        super().__init__()
        self.space_to_depth = SpaceToDepth(block_size=scale_factor)
        self.max_pool = nn.MaxPool2d(kernel_size=depth_block_size, stride=1, padding=depth_block_size // 2, return_indices=True)

    def forward(self, 
                depth: torch.Tensor, 
                jitter: torch.Tensor, 
                prev_jitter: torch.Tensor,
                motion: torch.Tensor, 
                prev_features: torch.Tensor, 
                prev_color: torch.Tensor) -> torch.Tensor:
        """
        motion: B, 2, H, W
        depth: B, 1, H, W
        jitter: B, 2, 1, 1
        prev_jitter: B, 2, 1, 1
            assumed that jitter is in relative coordinates 
            (i.e. -1 to 1, -1 is the whole image left, 1 is the whole image right)
        prev_features: B, 1, H, W
        prev_color: B, 3, H, W

        Note:
            H, W are target resolutions
            it is assumed that depth and motion are upsampled prior
        """

        # Jitter compensation for motion
        motion[:, 0] = motion[:, 0] + prev_jitter[:, 0] - jitter[:, 0] # x
        motion[:, 1] = motion[:, 1] + prev_jitter[:, 1] - jitter[:, 1] # y

        # Depth informed dilation
        # Get indices of closest pixels and use those motion vectors
        _, indices = self.max_pool(depth)
        motion = retrieve_elements_from_indices(motion, indices)

        # Warp previous features and color
        prev_features = warp(prev_features, motion)
        prev_color = warp(prev_color, motion)

        # Transform to input resolution
        prev_features = self.space_to_depth(prev_features)
        prev_color = self.space_to_depth(prev_color)

        return prev_features, prev_color


