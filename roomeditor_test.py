import os
import sys
import torch
import argparse
from CustomEvaluator import CustomEvaluator
from PIL import Image
sys.path.append("./dataset")
from dataset.dataset_train import TrainDataset

from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from models.referNet.pipeline_Roomeditor import Roomeditor_Pipeline, hack_attn_forward
from models.referNet.utils import get_attn_module
from models.mimicbrush.utils import is_torch2_available
if is_torch2_available():
    from models.mimicbrush.attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
else:
    from models.mimicbrush.attention_processor import AttnProcessor

from diffusers.schedulers import DDIMScheduler
from functools import partial

def remove_cross_attention(unet, mode):
    attn_modules = get_attn_module(unet, mode=mode)
    # Remove cross-attention modules (attn2 and norm2)
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

# Perform inference using the Diffusers pipeline.
@torch.no_grad()
def pipline_call_diffusers(
    batch,
    model_dict,
    generator,
    num_inference_steps,
    guidance_scale
):
    pipeline = model_dict["pipeline"]
    images = pipeline(
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        source_image=batch['ref_images'],
        image=batch['masked_background_images'],
        mask_image=batch['background_masks']
    ).images
    return images

def split_image(combined_image):
    """
    Splits a combined image (with two horizontally concatenated images) into two separate images.

    Args:
        combined_image (PIL.Image.Image): The combined image with dimensions (height: 512, width: 1024).

    Returns:
        tuple: Two PIL.Image.Image objects representing the split images.
    """
    # Ensure the input is a PIL image
    if not isinstance(combined_image, Image.Image):
        raise ValueError("Input must be a PIL Image object")

    # Get dimensions of the combined image
    width, height = combined_image.size

    # Validate the dimensions
    if height != 512 or width != 1024:
        raise ValueError("Expected combined image dimensions to be 512x1024")

    # Calculate the width of each split image
    single_width = width // 2

    # Split the image into two
    left_image = combined_image.crop((0, 0, single_width, height))
    right_image = combined_image.crop((single_width, 0, width, height))

    return left_image, right_image

def _collate_fn(batch):
    """
    Collate function for DataLoader.

    Args:
        batch: list of samples

    Returns:
        dict containing processed image pairs and metadata
    """
    ref_images = [sample['ref_image'] for sample in batch]
    ref_masks = [sample['ref_mask'] for sample in batch]
    
    background_images = [sample['background_image'] for sample in batch]
    masked_background_images = [sample['masked_background_image'] for sample in batch]
    background_masks = [sample['background_mask'] for sample in batch]

    # clip_image = clip_processor(ref_images, return_tensors="pt").pixel_values
    clip_images = ref_images  # Left for the image_encoder to process
    
    return {
        'id': [sample['id'] for sample in batch],
        'raw': [sample['raw'] for sample in batch],
        'clip_images': clip_images,
        'ref_images': ref_images,
        'ref_masks': ref_masks,
        'background_images': background_images,
        'masked_background_images': masked_background_images,
        'background_masks': background_masks
    }

# Process U-Net separately.
def set_precision_and_device(model_list, unet, precision, device):
    print(f"Setting precision to {precision} and device to {device}")
    for model in model_list:
        model.to(device=device, dtype=precision)
        
    unet.to(device=device, dtype=precision)
    if precision == torch.float16:
        # Specifically set attention processor for fp16
        unet.set_attn_processor(AttnProcessor())

def eval(args): 
    # Determine precision
    if args.precision == "fp32":
        precision = torch.float32
    elif args.precision == "fp16":
        precision = torch.float16
    elif args.precision == "bf16":
        precision = torch.bfloat16
    else:
        raise ValueError(f"Invalid precision: {args.precision}")
        
    # Import models
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_sd_inpainting_path,
        subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_sd_inpainting_path,
        subfolder="unet", in_channels=9,
    )
    # Remove cross-attention from U-Net
    remove_cross_attention(unet, mode='full')
    hack_attn_forward(unet)

    # Currently only supports loading the full U-Net checkpoint
    if os.path.exists(os.path.join(args.checkpoint_path, "unet")):
        print('=== found unet in checkpoint ===')
        from utils import load_safetensor
        ckpt = load_safetensor(os.path.join(
            args.checkpoint_path, "unet", "diffusion_pytorch_model.safetensors"
        ))
        unet.load_state_dict(ckpt)
    else:
        raise ValueError("unet not found in checkpoint!!!")
     
    pipeline = Roomeditor_Pipeline.from_pretrained(
        args.pretrained_sd_inpainting_path,
        vae=vae,
        unet=unet,
        safety_checker=None,
        feature_extractor=None
    )
    set_precision_and_device(
        [vae, pipeline],
        unet=unet, precision=precision, device="cuda"
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")

    # Dataset preparation
    dataset = TrainDataset(
        args.dataset_path,
        background_mask_color='black'
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=_collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model_dict = {
        "pipeline": pipeline,
    }
    evaluator = CustomEvaluator(
       checkpoint_path=args.checkpoint_path,
       save_path=args.save_path,
       dataset=dataset,
       dataloader=dataloader,
       seed=args.seed,
       model_dict=model_dict,
       postprocess=args.postprocess,
       pipeline_fn=pipline_call_diffusers
    )
    evaluator.evaluate(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation script for Stable Diffusion Inpainting"
    )
    parser.add_argument(
        "--pretrained_sd_inpainting_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Checkpoint path"
    )
    parser.add_argument(
        "--precision", type=str, default="fp32", help="Precision for model"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for dataloader"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1000,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random number generator"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=5.0, help="Guidance scale"
    )
    parser.add_argument(
        "--save_path", type=str, default="eval", help="Path to save the results"
    )
    parser.add_argument(
        "--postprocess", action="store_true", help="Postprocess the generated images"
    )
    args = parser.parse_args()

    eval(args)
