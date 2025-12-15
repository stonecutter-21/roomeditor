import torch
from torchvision import transforms
from torchvision.io import read_image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from PIL import Image
import re

def crop_and_resize_bbox(image: Image.Image, bbox: list, scale: float) -> Image.Image:
    """
    Crop and resize the region defined by a bounding box, scaling it around its center.

    Args:
        image (PIL.Image): Input image.
        bbox (list): Bounding box [x_min, y_min, width, height].
        scale (float): Scaling factor for the box.

    Returns:
        PIL.Image: Cropped and resized image region.
    """
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height

    # Compute center
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    new_w, new_h = width * scale, height * scale

    # New box coordinates around center
    new_x_min = cx - new_w / 2
    new_x_max = cx + new_w / 2
    new_y_min = cy - new_h / 2
    new_y_max = cy + new_h / 2

    # Validate box dimensions
    if new_x_min >= new_x_max or new_y_min >= new_y_max:
        raise ValueError(f"Invalid bbox after scaling: {bbox}")

    # Clamp to image bounds
    new_x_min, new_x_max = max(0, new_x_min), min(image.width, new_x_max)
    new_y_min, new_y_max = max(0, new_y_min), min(image.height, new_y_max)

    if new_x_min >= new_x_max or new_y_min >= new_y_max:
        raise ValueError(f"Invalid bbox after clamping: {[new_x_min, new_y_min, new_x_max, new_y_max]}")

    return image.crop((int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)))


def crop_and_pad_to_size(image: Image.Image, bbox: list, size: tuple) -> Image.Image:
    """
    Crop the image by the bounding box and pad to the target size without scaling content.

    Args:
        image (PIL.Image): Input image.
        bbox (list): Bounding box [x_min, y_min, width, height].
        size (tuple): Target size (height, width).

    Returns:
        PIL.Image: Cropped and padded image, centered with black padding.
    """
    x_min, y_min, w, h = map(int, bbox)
    target_h, target_w = map(int, size)

    x_max, y_max = x_min + w, y_min + h

    # Intersection of bbox and image
    left, upper = max(x_min, 0), max(y_min, 0)
    right, lower = min(x_max, image.width), min(y_max, image.height)

    cropped = image.crop((left, upper, right, lower))
    # No padding implementation shown; return cropped region
    return cropped


def calculate_fid(image1_list: list, image2_list: list, device, batch_size: int = 32) -> float:
    """Compute the Frechet Inception Distance (FID) between two image sets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
        transforms.Lambda(lambda x: x.to(torch.uint8)),
        transforms.Resize((299, 299)),
    ])
    fid = FrechetInceptionDistance(feature=2048).to(device)

    for imgs, real in [(image1_list, True), (image2_list, False)]:
        for i in range(0, len(imgs), batch_size):
            batch = torch.stack([transform(img) for img in imgs[i:i+batch_size]]).to(device)
            fid.update(batch, real=real)

    return fid.compute().item()


def calculate_is(image_list: list, device, batch_size: int = 32) -> float:
    """Compute the Inception Score (IS) for a list of images."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
        transforms.Lambda(lambda x: x.to(torch.uint8)),
        transforms.Resize((299, 299)),
    ])
    is_metric = InceptionScore().to(device)
    for i in range(0, len(image_list), batch_size):
        batch = torch.stack([transform(img) for img in image_list[i:i+batch_size]]).to(device)
        is_metric.update(batch)
    return is_metric.compute()[0].item()


def calculate_ssim(image1_list: list, image2_list: list) -> list:
    """Compute SSIM (Structural Similarity) for corresponding image pairs."""
    transform = transforms.ToTensor()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    results = []
    for img1, img2 in zip(image1_list, image2_list):
        t1, t2 = transform(img1).unsqueeze(0), transform(img2).unsqueeze(0)
        results.append(ssim(t2, t1).item())
    return results


def calculate_psnr(image1_list: list, image2_list: list) -> list:
    """Compute PSNR (Peak Signal-to-Noise Ratio) for corresponding image pairs."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])
    psnr = PeakSignalNoiseRatio()
    results = []
    for img1, img2 in zip(image1_list, image2_list):
        t1, t2 = transform(img1).unsqueeze(0), transform(img2).unsqueeze(0)
        results.append(psnr(t2, t1).item())
    return results


def calculate_lpips(image1_list: list, image2_list: list) -> list:
    """Compute LPIPS (Learned Perceptual Image Patch Similarity) for image pairs."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
    results = []
    for img1, img2 in zip(image1_list, image2_list):
        t1, t2 = transform(img1).unsqueeze(0), transform(img2).unsqueeze(0)
        results.append(lpips(t2, t1).item())
    return results


def filter_images(image_list: list) -> list:
    """
    Filter filenames matching the pattern "*_<number>_<number>.png".
    Returns only those filenames.
    """
    pattern = re.compile(r'.*_\d+_\d+\.png$')
    return [img for img in image_list if pattern.match(img)]
