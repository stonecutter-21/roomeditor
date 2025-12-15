import torch
from safetensors import safe_open
from diffusers.utils.torch_utils import is_compiled_module
from types import MethodType

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def get_params(model):
    return sum(p.numel() for p in model.parameters())

def load_safetensor(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def change_forward_fn(module, forward_fn):
    module._original_inner_forward = module.forward
    module.forward = MethodType(forward_fn, module)


# Get U-Net's attention modules in the correct order
def get_attn_module_for_correct_order(unet, mode):
    unet = unet.module if hasattr(unet, 'module') else unet
    down_modules = []
    for down_block in unet.down_blocks:
        if hasattr(down_block, 'attentions'):
            down_modules.extend(x.transformer_blocks[0] for x in down_block.attentions)
    mid_modules = []
    mid_modules.extend((x.transformer_blocks[0] for x in unet.mid_block.attentions))
    up_modules = []
    for up_block in unet.up_blocks:
        if hasattr(up_block, 'attentions'):
            up_modules.extend(x.transformer_blocks[0] for x in up_block.attentions)
            
    if mode == 'midup':
        return mid_modules + up_modules
    else:
        raise ValueError(f"Invalid mode {mode}.")


# Assign names to each U-Net attention block for easier later retrieval
def name_unet_attn_blocks(unet):
    def _name_attn_blocks(unet, mode):
        blocks = get_attn_module_for_correct_order(unet, mode)
        for i, attn_block in enumerate(blocks):
            attn_block.module_name = f"{mode}_{i}"
    _name_attn_blocks(unet, 'down')
    _name_attn_blocks(unet, 'mid')
    _name_attn_blocks(unet, 'up')
