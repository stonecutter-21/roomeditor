from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F

from pycocotools import mask as mask_utils

# Convert a PNG mask to RLE format
def mask_to_rle(image: Image.Image) -> dict:
    """
    Convert a mask image to RLE format.

    Parameters:
        image (PIL.Image): Input mask image in RGB format.
    Returns:
        dict: RLE segmentation in COCO format.
    """
    # Convert the image to a numpy array (grayscale)
    mask = np.array(image.convert("L"))

    # Ensure binary mask (assuming non-zero values indicate the mask)
    binary_mask = (mask > 0).astype(np.uint8)

    # Encode to RLE
    rle = mask_utils.encode(np.asfortranarray(binary_mask))

    # Convert RLE to COCO format (decode counts for compatibility)
    rle['counts'] = rle['counts'].decode('utf-8')

    return rle

def seg2mask(seg, h, w):
    if isinstance(seg, dict):
        # Indicates it's already in RLE format
        assert seg['size'] == [h, w], f"rle size {seg['size']} != {h, w}"
        # For uncompressed RLE format, need to convert
        if isinstance(seg['counts'], list):
            rle = mask_utils.frPyObjects([seg], h, w)
        else:
            rle = seg
    else:
        rle = mask_utils.frPyObjects(seg, h, w)
    mask = mask_utils.decode(rle)
    return mask[:, :, 0] if mask.ndim == 3 else mask

def mask2pil(mask):
    mask = np.array(mask)
    mask = np.stack([mask, mask, mask], axis=-1) * 255
    return Image.fromarray(mask)

def pil2mask(pil_img):
    return np.array(pil_img)[:, :, 0]

def pad_img_to_square(original_image, is_mask=False):
    width, height = original_image.size
    if height == width:
        return original_image

    # Determine padding to make the image square
    if height > width:
        padding = (height - width) // 2
        new_size = (height, height)
    else:
        padding = (width - height) // 2
        new_size = (width, width)

    # Create a new blank image: black for masks, white for images
    fill_color = "black" if is_mask else "white"
    new_image = Image.new("RGB", new_size, fill_color)

    # Paste the original image centered
    if height > width:
        new_image.paste(original_image, (padding, 0))
    else:
        new_image.paste(original_image, (0, padding))
    return new_image

# Previous version, now deprecated, kept for reference
def collage_region(low, high, mask):
    # Binarize the mask
    mask = (np.array(mask) > 128).astype(np.uint8)
    low = np.array(low).astype(np.uint8)
    low = (low * 0).astype(np.uint8)
    high = np.array(high).astype(np.uint8)
    mask_3 = mask
    collage = low * mask_3 + high * (1 - mask_3)
    return Image.fromarray(collage)

def apply_mask(image, mask, mode='background_g'):
    """
    mode: mask mode—
      'product' (white fill for masked areas, mask black as cut-out),
      'background_b' (black fill for masked areas, mask white as cut-out),
      'background_g' (gray fill for masked areas, mask white as cut-out)
    """
    if mode == 'product':
        return _apply_mask(image, mask, fill_color='white', mask_color='black')
    elif mode == 'background_b':
        return _apply_mask(image, mask, fill_color='black', mask_color='white')
    elif mode == 'background_g':
        return _apply_mask(image, mask, fill_color='gray', mask_color='white')
    else:
        raise ValueError("mode must be 'product', 'background_b' or 'background_g'")

def _apply_mask(image: Image.Image, mask: Image.Image, fill_color: str = 'black', mask_color: str = 'white') -> Image.Image:
    """
    Apply a mask to an image.

    :param image: Original image (PIL Image)
    :param mask: Mask image (PIL Image), black areas indicate background, white areas indicate masked region
    :param fill_color: Fill color, one of 'black', 'white', or 'gray'
    :param mask_color: Which region of mask to apply: 'black' or 'white'
    :return: Masked image (PIL Image)
    """
    image_np = np.array(image)
    mask_np = np.array(mask)

    # Determine fill value
    if fill_color == 'black':
        fill_value = 0
    elif fill_color == 'white':
        fill_value = 255
    elif fill_color == 'gray':
        fill_value = 128
    else:
        raise ValueError("fill_color must be 'black', 'white', or 'gray'")

    # Determine which mask region to use
    if mask_color == 'white':
        mask_region = mask_np == 255
    elif mask_color == 'black':
        mask_region = mask_np == 0
    else:
        raise ValueError("mask_color must be 'black' or 'white'")

    # Fill the specified regions
    image_np[mask_region] = fill_value

    return Image.fromarray(image_np)

def crop_image_by_mask(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    """
    Crop an image based on the non-zero region of a mask and return both the cropped image and corresponding mask.

    Parameters:
        image (PIL.Image): The original image.
        mask (PIL.Image): The mask image in RGB format.

    Returns:
        tuple[PIL.Image, PIL.Image]: The cropped image and the corresponding cropped mask.
    """
    mask_gray = mask.convert("L")
    bbox = mask_gray.getbbox()
    if bbox is None:
        raise ValueError("Mask does not contain any non-zero regions.")
    cropped_image = image.crop(bbox)
    cropped_mask = mask.crop(bbox)
    return cropped_image, cropped_mask

def resize_to_fit_mask(reference_image: Image.Image, reference_mask: Image.Image, target_mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    """
    Resize the reference image and its mask to fit within the non-zero region of the target mask.

    Parameters:
        reference_image (PIL.Image): The cropped reference image.
        reference_mask (PIL.Image): The corresponding cropped reference mask.
        target_mask (PIL.Image): The target mask to fit the image and mask into.

    Returns:
        tuple[PIL.Image, PIL.Image]: The resized image and resized mask that fit within the target mask.
    """
    target_mask_gray = target_mask.convert("L")
    bbox = target_mask_gray.getbbox()
    if bbox is None:
        raise ValueError("Target mask does not contain any non-zero regions.")

    target_width = bbox[2] - bbox[0]
    target_height = bbox[3] - bbox[1]

    resized_image = reference_image.resize((target_width, target_height), Image.LANCZOS)
    resized_mask = reference_mask.resize((target_width, target_height), Image.NEAREST)

    result_image = Image.new("RGB", target_mask.size, (255, 255, 255))
    result_mask = Image.new("L", target_mask.size, 0)

    result_image.paste(resized_image, bbox[:2])
    result_mask.paste(resized_mask, bbox[:2])

    return result_image, result_mask

def background_image_fusion(background: Image.Image, background_mask: Image.Image,
                            reference: Image.Image, reference_mask: Image.Image,
                            alpha=0.5) -> Image.Image:
    """
    Crop the reference image according to its mask, then blend it into the background image.

    Parameters:
        background (PIL.Image): Background image.
        background_mask (PIL.Image): Binary mask of the background image.
        reference (PIL.Image): Reference image.
        reference_mask (PIL.Image): Binary mask of the reference image.
        alpha (float): Fusion weight in the range [0, 1].

    Returns:
        PIL.Image: Fused RGB image.
    """
    # Ensure masks are binary
    if background_mask.mode != '1':
        background_mask = background_mask.convert('1')
    if reference_mask.mode != '1':
        reference_mask = reference_mask.convert('1')

    # Ensure images are in RGB mode
    if background.mode != 'RGB':
        background = background.convert('RGB')
    if reference.mode != 'RGB':
        reference = reference.convert('RGB')

    # Get the bounding box of the reference mask
    ref_bbox = reference_mask.getbbox()
    if not ref_bbox:
        return background  # If reference mask is empty, return background

    # Crop the reference image and mask
    cropped_reference = reference.crop(ref_bbox)
    cropped_reference_mask = reference_mask.crop(ref_bbox)

    # Get the bounding box of the background mask
    bg_bbox = background_mask.getbbox()
    if not bg_bbox:
        return background  # If background mask is empty, return background

    # Compute the size of the background mask region
    bg_width = bg_bbox[2] - bg_bbox[0]
    bg_height = bg_bbox[3] - bg_bbox[1]

    # Resize the cropped reference to fit the background mask region
    resized_reference = cropped_reference.resize((bg_width, bg_height))
    resized_reference_mask = cropped_reference_mask.resize((bg_width, bg_height))

    # Create a copy of the background for the result
    result = background.copy()

    # Extract the background region to blend into
    bg_region = result.crop(bg_bbox)

    # Blend using PIL's blend function (invert alpha because blend uses second image weight as 1-alpha)
    blend_result = Image.blend(resized_reference, bg_region, 1 - alpha)

    # Create a mask for the composite region
    mask_for_composite = Image.new('L', (bg_width, bg_height), 0)
    if resized_reference_mask.mode != 'L':
        resized_reference_mask = resized_reference_mask.convert('L')
    mask_for_composite.paste(resized_reference_mask)

    # Composite the blended region over the background using the mask
    composited_region = Image.composite(blend_result, bg_region, mask_for_composite)

    # Paste the composite region back into the result image
    result.paste(composited_region, bg_bbox)

    # Ensure the result is in RGB mode
    if result.mode != 'RGB':
        result = result.convert('RGB')

    return result
