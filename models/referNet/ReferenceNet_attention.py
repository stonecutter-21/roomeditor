# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py

import torch
import torch.nn.functional as F
import random
from einops import rearrange
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from types import MethodType
from functools import partial
from diffusers.models.attention_processor import Attention 

from models.referNet.utils import get_attn_module, get_mask


def remove_cross_attention(unet, mode):
    attn_modules = get_attn_module(unet, mode=mode)
    # Remove cross-attention components (attn2 and norm2) from each module
    for attn_module in attn_modules:
        # Remove methods
        attn_module.attn2 = None
        attn_module.norm2 = None
        # Remove corresponding weights
        state_dict = attn_module.state_dict()
        for key in list(state_dict.keys()):
            if 'attn2' in key or 'norm2' in key:
                del state_dict[key]
        attn_module.load_state_dict(state_dict)


# Rewrite second half of the Transformer forward function
# Based on debugging, only basic normalization and feed-forward are needed
def forward_second_half(
    hidden_states, encoder_hidden_states, encoder_attention_mask, cross_attention_kwargs,
    attn2, norm2, norm3, ff
):
    if attn2 is not None:
        # Cross-attention phase
        norm_hidden_states = norm2(hidden_states)
        attn_output = attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        hidden_states = attn_output + hidden_states

    # Feed-forward phase
    norm_hidden_states = norm3(hidden_states)
    ff_output = ff(norm_hidden_states)
    hidden_states = ff_output + hidden_states

    return hidden_states


def hacked_basic_transformer_forward_write(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[torch.LongTensor] = None,
    save_features: bool = False,
):
    assert not self.use_ada_layer_norm, "use_ada_layer_norm not supported"
    assert not self.use_ada_layer_norm_zero, "use_ada_layer_norm_zero not supported"
    assert not self.only_cross_attention, "only_cross_attention not supported"
    
    norm_hidden_states = self.norm1(hidden_states)
    cross_attention_kwargs = cross_attention_kwargs or {}

    # Self-attention write phase: append normalized states to bank
    self.bank.append(norm_hidden_states.clone())
    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )
    if save_features:
        self.hidden_input = norm_hidden_states.clone()
        self.attn_output = attn_output.clone()
        
    hidden_states = attn_output + hidden_states
    hidden_states = forward_second_half(
        hidden_states, encoder_hidden_states, encoder_attention_mask, cross_attention_kwargs,
        self.attn2, self.norm2, self.norm3, self.ff
    )
    return hidden_states


def hacked_basic_transformer_forward_read(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[torch.LongTensor] = None,
    do_classifier_free_guidance: bool = False,
    drop_during_training: bool = False,
    save_features: bool = False
):
    assert not self.use_ada_layer_norm, "use_ada_layer_norm not supported"
    assert not self.use_ada_layer_norm_zero, "use_ada_layer_norm_zero not supported"
    assert not self.only_cross_attention, "only_cross_attention not supported"
    
    norm_hidden_states = self.norm1(hidden_states)
    cross_attention_kwargs = cross_attention_kwargs or {}

    # Fuse current hidden states with stored bank entries
    modify_norm_hidden_states = torch.cat([norm_hidden_states] + self.bank, dim=1)
    
    # Unconditional self-attention with fusion input
    hidden_states_uc = self.attn1(
        modify_norm_hidden_states, 
        encoder_hidden_states=modify_norm_hidden_states,
        attention_mask=attention_mask
    )[:, :hidden_states.shape[-2], :]
    
    # Raw self-attention without fusion
    hidden_states_raw = self.attn1(
        norm_hidden_states, 
        encoder_hidden_states=norm_hidden_states,
        attention_mask=attention_mask
    )

    # Combine fused and raw outputs (ratio set to 1.0 for full fusion)
    ratio = 1.0
    hidden_states_uc = hidden_states_uc * ratio + hidden_states_raw * (1 - ratio)
    if save_features:
        self.hidden_input = norm_hidden_states.clone()
        self.attn_output = hidden_states_uc.clone()
    hidden_states_uc = hidden_states_uc + hidden_states 

    # Apply classifier-free guidance mask if specified
    hidden_states_c = hidden_states_uc.clone()
    device = hidden_states.device
    _uc_mask = get_mask(do_classifier_free_guidance, drop_during_training, hidden_states.shape[0], device)
    if _uc_mask.any():
        guided = self.attn1(
            norm_hidden_states[_uc_mask],
            encoder_hidden_states=norm_hidden_states[_uc_mask],
            attention_mask=attention_mask
        ) + hidden_states[_uc_mask]
        hidden_states_c[_uc_mask] = guided
      
    hidden_states = hidden_states_c.clone()
    hidden_states = forward_second_half(
        hidden_states, encoder_hidden_states, encoder_attention_mask, cross_attention_kwargs,
        self.attn2, self.norm2, self.norm3, self.ff
    )
    return hidden_states


def hacked_adapter_forward_read(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[torch.LongTensor] = None,
    do_classifier_free_guidance: bool = False,
    drop_during_training: bool = False,
):
    assert not self.use_ada_layer_norm, "use_ada_layer_norm not supported"
    assert not self.use_ada_layer_norm_zero, "use_ada_layer_norm_zero not supported"
    assert not self.only_cross_attention, "only_cross_attention not supported"
    
    # Adapter phase: update hidden_states with reference feature fusion
    hidden_states = self.adapter(
        hidden_states,
        self.bank,
        do_classifier_free_guidance,
        drop_during_training
    )
    
    # Proceed with original forward logic
    norm_hidden_states = self.norm1(hidden_states)
    cross_attention_kwargs = cross_attention_kwargs or {}

    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )
    hidden_states = attn_output + hidden_states
    hidden_states = forward_second_half(
        hidden_states, encoder_hidden_states, encoder_attention_mask, cross_attention_kwargs,
        self.attn2, self.norm2, self.norm3, self.ff
    ) 
    return hidden_states


class ReferenceNetAttention:
    def __init__(self, 
        unet,
        mode,
        do_classifier_free_guidance,
        drop_during_training=False,
        attention_auto_machine_weight: float = float('inf'),
        gn_auto_machine_weight: float = 1.0,
        style_fidelity: float = 1.0,
        reference_attn=True,
        fusion_blocks="midup",
        forward_choice='default',
        save_features=False,
    ) -> None:
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.fusion_blocks = fusion_blocks
        self.forward_choice = forward_choice
        self.save_features = save_features
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            drop_during_training,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            fusion_blocks,
            save_features
        )

    def register_reference_hooks(
        self, 
        mode,
        do_classifier_free_guidance,
        drop_during_training,
        attention_auto_machine_weight,
        gn_auto_machine_weight,
        style_fidelity,
        reference_attn,
        fusion_blocks='midup',
        save_features=False,
        dtype=torch.float32,
    ):
        if not self.reference_attn:
            return None
        
        unet = self.unet.module if hasattr(self.unet, 'module') else self.unet
        attn_modules = get_attn_module(unet, mode=fusion_blocks)
        for i, module in enumerate(attn_modules):
            module._original_inner_forward = module.forward
            if mode == 'write':
                forward_fn = partial(
                    hacked_basic_transformer_forward_write,
                    save_features=save_features
                )
                module.forward = MethodType(forward_fn, module)
            elif mode == 'read':
                if self.forward_choice == 'default':
                    forward_fn = hacked_basic_transformer_forward_read
                elif self.forward_choice == 'adapter':
                    forward_fn = hacked_adapter_forward_read
                else:
                    raise ValueError(f"forward_choice {self.forward_choice} not supported")
                partial_forward = partial(
                    forward_fn, 
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    drop_during_training=drop_during_training,
                    save_features=save_features
                )
                module.forward = MethodType(partial_forward, module)
            module.bank = []
            module.attn_weight = float(i) / float(len(attn_modules))
    
    def update(self, writer, dtype=torch.float32):
        if not self.reference_attn:
            return None
        
        unet = self.unet.module if hasattr(self.unet, 'module') else self.unet
        writer_unet = writer.unet.module if hasattr(writer.unet, 'module') else writer.unet
        reader_attn_modules = get_attn_module(unet, mode=self.fusion_blocks)
        writer_attn_modules = get_attn_module(writer_unet, mode=self.fusion_blocks)
            
        assert reader_attn_modules, "reader_attn_modules is empty"
        assert writer_attn_modules, "writer_attn_modules is empty"
            
        for r, w in zip(reader_attn_modules, writer_attn_modules):
            r.bank = [v.clone().to(dtype) for v in w.bank]
    
    def clear(self):
        if not self.reference_attn:
            return None
        unet = self.unet.module if hasattr(self.unet, 'module') else self.unet
        reader_attn_modules = get_attn_module(unet, mode=self.fusion_blocks)
        for r in reader_attn_modules:
            if hasattr(r, 'bank'):
                r.bank.clear()
            else:
                print(f"WARNING: bank not found in {r}")



