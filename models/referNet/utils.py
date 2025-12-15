import torch
from diffusers.models.attention import BasicTransformerBlock
from ..mimicbrush.attention import BasicTransformerBlock as _BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.downsampling import Downsample2D
from diffusers.models.upsampling import Upsample2D
import random

# Basic depth-first search of all submodules
def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

# Note: These mode-based extractors could be unified by a helper function
def get_attn_module(unet, mode="midup"):
    if mode == "down":
        attn_modules = [
            module for module in torch_dfs(unet.down_blocks) 
            if isinstance(module, BasicTransformerBlock) or isinstance(module, _BasicTransformerBlock)
        ]
    elif mode == "mid":
        attn_modules = [
            module for module in torch_dfs(unet.mid_block) 
            if isinstance(module, BasicTransformerBlock) or isinstance(module, _BasicTransformerBlock)
        ]
    elif mode == "up":
        attn_modules = [
            module for module in torch_dfs(unet.up_blocks) 
            if isinstance(module, BasicTransformerBlock) or isinstance(module, _BasicTransformerBlock)
        ]
    elif mode == "midup":
        attn_modules = [
            module for module in (torch_dfs(unet.mid_block) + torch_dfs(unet.up_blocks)) 
            if isinstance(module, BasicTransformerBlock) or isinstance(module, _BasicTransformerBlock)
        ]
    elif mode == "full":
        attn_modules = [
            module for module in torch_dfs(unet) 
            if isinstance(module, BasicTransformerBlock) or isinstance(module, _BasicTransformerBlock)
        ]
    else:
        raise ValueError(f"mode {mode} not supported")
    # Sort by decreasing feature dimension
    attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
    return attn_modules

def get_resnet_module(unet, mode='full'):
    if mode == 'down':
        resnet_modules = [
            module for module in torch_dfs(unet.down_blocks) 
            if isinstance(module, ResnetBlock2D)
        ]
    elif mode == 'mid':
        resnet_modules = [
            module for module in torch_dfs(unet.mid_block) 
            if isinstance(module, ResnetBlock2D)
        ]
    elif mode == 'up':
        resnet_modules = [
            module for module in torch_dfs(unet.up_blocks) 
            if isinstance(module, ResnetBlock2D)
        ]
    elif mode == 'full':
        resnet_modules = [
            module for module in torch_dfs(unet) 
            if isinstance(module, ResnetBlock2D)
        ]
    else:
        raise ValueError(f"mode {mode} not supported")
    return resnet_modules

def get_downsample_module(unet, mode):
    if mode == 'down':
        downsample_modules = [
            module for module in torch_dfs(unet.down_blocks) 
            if isinstance(module, Downsample2D)
        ]
    elif mode == 'mid':
        downsample_modules = [
            module for module in torch_dfs(unet.mid_block) 
            if isinstance(module, Downsample2D)
        ]
    elif mode == 'up':
        downsample_modules = [
            module for module in torch_dfs(unet.up_blocks) 
            if isinstance(module, Downsample2D)
        ]
    elif mode == 'full':
        downsample_modules = [
            module for module in torch_dfs(unet) 
            if isinstance(module, Downsample2D)
        ]
    else:
        raise ValueError(f"mode {mode} not supported")
    return downsample_modules

def get_upsample_module(unet, mode):
    if mode == 'down':
        upsample_modules = [
            module for module in torch_dfs(unet.down_blocks) 
            if isinstance(module, Upsample2D)
        ]
    elif mode == 'mid':
        upsample_modules = [
            module for module in torch_dfs(unet.mid_block) 
            if isinstance(module, Upsample2D)
        ]
    elif mode == 'up':
        upsample_modules = [
            module for module in torch_dfs(unet.up_blocks) 
            if isinstance(module, Upsample2D)
        ]
    elif mode == 'full':
        upsample_modules = [
            module for module in torch_dfs(unet) 
            if isinstance(module, Upsample2D)
        ]
    else:
        raise ValueError(f"mode {mode} not supported")
    return upsample_modules

def calc_mean_std(feat, eps: float = 1e-5):
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std

def get_mask(
    do_classifier_free_guidance, drop_during_training,
    batch_size,
    device
):
    """
    Unified mask generation for both classifier-free guidance and training dropout.
    Maintains original training logic for drop_during_training.
    
    Args:
        do_classifier_free_guidance (bool): Whether to use classifier-free guidance
        drop_during_training (bool): Whether to apply dropout during training
        batch_size (int): Batch size
        device (torch.device): Device for the mask tensor
    
    Returns:
        torch.Tensor: Boolean mask tensor of shape (batch_size,)
    """
    assert not (do_classifier_free_guidance and drop_during_training), "Cannot use both guidance and dropout simultaneously"
    if do_classifier_free_guidance:
        # Classifier-free guidance: first half True, second half False
        mask_index = [1] * (batch_size // 2) + [0] * (batch_size - batch_size // 2)
        mask = torch.tensor(mask_index, device=device, dtype=torch.bool)
    elif drop_during_training:
        # Original dropout logic during training (25% drop rate)
        mask_index = [0 for _ in range(batch_size)]
        for i in range(int(batch_size * 0.25)):
            mask_index[i] = 1
        mask = torch.tensor(mask_index, device=device, dtype=torch.bool)
        # If no positions dropped, randomly drop all with 25% chance
        if mask.sum() == 0 and random.random() < 0.25:
            mask[:] = True
    else:
        mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    return mask
