import gradio as gr
import cv2
import os
from PIL import Image
import numpy as np

import dataset.data_utils as data_utils
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from models.referNet.ReferenceNet_attention import remove_cross_attention
from models.referNet.pipeline_Roomeditor import Roomeditor_Pipeline, hack_attn_forward
from diffusers.schedulers import DDIMScheduler
import torch
from models.mimicbrush.utils import is_torch2_available

if is_torch2_available():
    from models.mimicbrush.attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
else:
    from models.mimicbrush.attention_processor import AttnProcessor

# Separate handling for the U-Net component
def set_precision_and_device(model_list, unet, precision, device):
    print(f"Setting precision to {precision} and device to {device}")
    for model in model_list:
        model.to(device=device, dtype=precision)
        
    unet.to(device=device, dtype=precision)
    if precision == torch.float16:
        unet.set_attn_processor(AttnProcessor())  # Intended specifically for fp16 support


# Load models
pretrained_sd_inpainting_path = "" # NOTE: Download from https://huggingface.co/runwayml/stable-diffusion-inpainting/
checkpoint_path = "" # Note: Down from https://share.multcloud.link/share/74f20132-4269-41d0-8355-0c39106de6b0
precision = torch.float16

vae = AutoencoderKL.from_pretrained(
    pretrained_sd_inpainting_path,
    subfolder="vae"
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_sd_inpainting_path,
    subfolder="unet",
    in_channels=9,
)

# Reload U-Net from checkpoint
# Remove cross-attention layers
remove_cross_attention(unet, mode='full')
hack_attn_forward(unet)

# Only supports full U-Net checkpoints at the moment
if os.path.exists(os.path.join(checkpoint_path, "unet")):
    print('=== found unet in checkpoint ===')
    from utils import load_safetensor
    ckpt = load_safetensor(os.path.join(checkpoint_path, "unet", "diffusion_pytorch_model.safetensors"))
    unet.load_state_dict(ckpt)
else:
    raise ValueError("unet not found in checkpoint!!!")
    

pipeline = Roomeditor_Pipeline.from_pretrained(
    pretrained_sd_inpainting_path,
    vae=vae,
    unet=unet,
    safety_checker=None,
    feature_extractor=None
)
set_precision_and_device(
    [vae, pipeline],
    unet=unet,
    precision=precision,
    device="cuda"
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to("cuda")

def get_mask_from_rgba(rgba_image):
    """
    Input:  PIL.Image in RGBA mode.
    Output: RGB-mode mask image where non-transparent areas are white and transparent areas are black.
    """
    # Ensure image is in RGBA mode
    if rgba_image.mode != "RGBA":
        raise ValueError("Input image is not in RGBA mode")
        # rgba_image = rgba_image.convert("RGBA")
    
    # Extract the alpha channel
    alpha = rgba_image.getchannel("A")
    
    # Create a binary mask from the alpha channel: non-transparent = 255, transparent = 0
    binary_mask = alpha.point(lambda p: 255 if p > 0 else 0)
    
    # Convert single-channel mask to RGB by replicating across three channels
    rgb_mask = Image.merge("RGB", (binary_mask, binary_mask, binary_mask))
    
    return rgb_mask

def data_pre_process(back_image, back_mask, ref_image, ref_mask):
    # 1. Pad images to square
    back_image = data_utils.pad_img_to_square(back_image)
    back_mask = data_utils.pad_img_to_square(back_mask, is_mask=True)
    ref_image = data_utils.pad_img_to_square(ref_image)
    ref_mask = data_utils.pad_img_to_square(ref_mask, is_mask=True)
    
    # 2. Apply mask
    # If ref_mask is all black, there's no mask; use ref_image directly
    if np.all(np.array(ref_mask) == 0):
        ref_image = ref_image
    else:
        ref_image = data_utils.apply_mask(ref_image, ref_mask, mode='product')
    back_image = data_utils.apply_mask(back_image, back_mask, mode='background_b')

    return back_image, back_mask, ref_image, ref_mask

def data_post_process(gen_image, back_image, back_mask):
    # 1. Crop and resize
    gen_image = np.array(gen_image)
    gen_image = crop_padding_and_resize(back_image.size[1], back_image.size[0], gen_image)
    gen_image = Image.fromarray(gen_image)
    # 2. Blend with the original background
    gen_image = retrain_background(gen_image, back_image, back_mask)
    return gen_image

def crop_padding_and_resize(ori_height, ori_width, square_image):
    scale = max(ori_height / square_image.shape[0], ori_width / square_image.shape[1])
    resized_square_image = cv2.resize(
        square_image,
        (int(square_image.shape[1] * scale), int(square_image.shape[0] * scale))
    )
    padding_size = max(
        resized_square_image.shape[0] - ori_height,
        resized_square_image.shape[1] - ori_width
    )
    if ori_height < ori_width:
        top = padding_size // 2
        bottom = resized_square_image.shape[0] - (padding_size - top)
        cropped_image = resized_square_image[top:bottom, :, :]
    else:
        left = padding_size // 2
        right = resized_square_image.shape[1] - (padding_size - left)
        cropped_image = resized_square_image[:, left:right, :]
    return cropped_image

def retrain_background(gen_image, raw_image, mask):
    # Convert images to numpy arrays
    raw_image = np.array(raw_image).astype(np.uint8)
    mask = np.array(mask)
    gen_image = np.array(gen_image).astype(np.uint8)
    # Apply Gaussian blur to mask edges
    for i in range(10):
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask = mask / 255
    # Compose generated image over raw background
    gen_image = gen_image * mask + raw_image * (1 - mask)
    gen_image = gen_image.astype(np.uint8)
    gen_image = Image.fromarray(gen_image)
    return gen_image

def zoom_in_background(back_image, back_mask):
    """
    Centered at the bounding box center of the mask's white area, crop the image and mask
    using twice the bounding box side length.

    Args:
        back_image (PIL.Image): Original RGB image.
        back_mask (PIL.Image): RGB mask image where the target region is white.

    Returns:
        tuple: (cropped_image, cropped_mask, metadata_dict).
    """
    # Get the bounding box of the mask
    bbox = back_mask.getbbox()
    if not bbox:
        raise ValueError("No non-zero pixels found in mask")
    
    # Parse bounding box coordinates
    min_x, min_y, max_x, max_y = bbox
    
    # Compute bounding box center
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    
    # Compute side length of the bounding box
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    bbox_length = max(bbox_width, bbox_height)
    
    # Calculate new crop side length (2.5 times the bbox length)
    new_length = int(bbox_length * 2.5)
    
    # Determine cropping coordinates
    crop_x1 = max(0, center_x - new_length // 2)
    crop_y1 = max(0, center_y - new_length // 2)
    crop_x2 = min(back_image.width, crop_x1 + new_length)
    paste_y2 = min(back_image.height, crop_y1 + new_length)
    
    # Crop image and mask
    cropped_image = back_image.crop((crop_x1, crop_y1, crop_x2, paste_y2))
    cropped_mask = back_mask.crop((crop_x1, crop_y1, crop_x2, paste_y2))
    
    # Return metadata dictionary
    zoom_in_meta = {
        "back_image": back_image.copy(),
        "back_mask": back_mask.copy(),
        "center_point": (center_x, center_y),
        "length": new_length
    }
    
    return cropped_image, cropped_mask, zoom_in_meta

def zoom_out(generated_image, zoom_in_meta):
    """
    Paste the generated image back into its original position in the source image.

    Args:
        generated_image (PIL.Image): Generated image patch.
        zoom_in_meta (dict): Metadata dictionary returned by zoom_in_background.

    Returns:
        PIL.Image: Full image after pasting the generated patch.
    """
    # Retrieve original image and parameters from metadata
    original_image = zoom_in_meta["back_image"]
    center_x, center_y = zoom_in_meta["center_point"]
    length = zoom_in_meta["length"]
    
    # Compute paste coordinates (same as crop location in zoom_in)
    paste_x1 = max(0, center_x - length // 2)
    paste_y1 = max(0, center_y - length // 2)
    paste_x2 = min(original_image.width, paste_x1 + length)
    paste_y2 = min(original_image.height, paste_y1 + length)
    
    # Create a copy of the original image to avoid modifying it
    result_image = original_image.copy()
    
    # Paste the generated image at the computed position
    result_image.paste(generated_image, (paste_x1, paste_y1))
    
    return result_image

def single_inference(
    back_image, back_mask, ref_image, ref_mask,
    ddim_steps, scale, seed, zoom_in=False
):
    if zoom_in:
        back_image, back_mask, zoom_in_meta = zoom_in_background(back_image, back_mask)
    old_back_image = back_image.copy()
    old_back_mask = back_mask.copy()
    back_image, back_mask, ref_image, ref_mask = data_pre_process(
        back_image, back_mask, ref_image, ref_mask
    )
    
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipeline(
        guidance_scale=scale,
        num_inference_steps=ddim_steps,
        generator=generator,
        source_image=[ref_image],
        image=[back_image],
        mask_image=[back_mask]
    ).images[0]
    image = data_post_process(image, old_back_image, old_back_mask)
    if zoom_in:
        image = zoom_out(image, zoom_in_meta)
    return image

def run_local(base, ref, ddim_steps, scale, seed, zoom_in):
    # Data preprocessing
    back_image = base['background'].convert('RGB')
    back_mask = get_mask_from_rgba(base['layers'][0])
    ref_image = ref['background'].convert('RGB')
    ref_mask = get_mask_from_rgba(ref['layers'][0])
    # Inference
    image = single_inference(
        back_image, back_mask, ref_image, ref_mask,
        ddim_steps, scale, seed, zoom_in=zoom_in
    )
    return [image]

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("# Demo ")
        with gr.Row():
            baseline_gallery = gr.Gallery(
                label='Output', show_label=True, elem_id="gallery", columns=1, height=768
            )
            with gr.Accordion("Advanced Option", open=True):
                num_samples = 1
                ddim_steps = gr.Slider(
                    label="Steps", minimum=1, maximum=1000, value=50, step=1
                )
                scale = gr.Slider(
                    label="Guidance Scale", minimum=0, maximum=20, value=2.0, step=0.1
                )
                seed = gr.Slider(
                    label="Seed", minimum=-1, maximum=1000, step=1, value=-1
                )
                zoom_in = gr.Checkbox(label="Zoom In")

        gr.Markdown("# Upload the source image and reference image")
        gr.Markdown("### Tips: you could adjust the brush size")

        with gr.Row():
            base = gr.ImageEditor(
                label="Source",
                type="pil",
                brush=gr.Brush(
                    colors=["#000000"],
                    default_size=30,
                    color_mode="fixed"
                ),
                layers=False,
                interactive=True
            )
            ref = gr.ImageEditor(
                label="Reference",
                type="pil",
                brush=gr.Brush(
                    colors=["#000000"],
                    default_size=30,
                    color_mode="fixed"
                ),
                layers=False,
                interactive=True
            )
        run_local_button = gr.Button(value="Run")

    run_local_button.click(
        fn=run_local,
        inputs=[base, ref, ddim_steps, scale, seed, zoom_in],
        outputs=[baseline_gallery]
    )

demo.launch(server_name="0.0.0.0")
