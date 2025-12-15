from typing import Dict
import numpy as np
from PIL import Image
import dataset.data_utils as data_utils
from diffusers.image_processor import VaeImageProcessor
import cv2
import math
import random

import albumentations as A


class ImageMaskAugmenter:
    """
    A class for synchronized image and mask augmentation using albumentations library.
    """
    def __init__(
        self, 
        flip_prob=0.5,
        rotation_prob=0.5,
        max_rotation_angle=30,
        scale_prob=0.3,
        scale_range=(0.8, 1.2),
        min_crop_ratio=0.5,
        seed=42
    ):
        """
        Initialize the augmenter with transformation probabilities and parameters.
        
        Args:
            flip_prob (float): Probability of horizontal flip (0-1)
            rotation_prob (float): Probability of rotation (0-1)
            max_rotation_angle (int): Maximum rotation angle in degrees
            scale_prob (float): Probability of scaling (0-1)
            scale_range (tuple): Range of scaling factors (min, max)
            min_crop_ratio (float): Minimum ratio of crop size to image size (0-1)
        """
        self.seed = seed
        self.min_crop_ratio = min_crop_ratio
        self.transform = A.Compose([
            # Scaling
            A.RandomScale(
                scale_limit=(scale_range[0] - 1, scale_range[1] - 1),
                p=scale_prob
            ),
            # Rotation (auto-crop black borders)
            A.Rotate(
                limit=max_rotation_angle,
                p=rotation_prob,
                border_mode=0,
                crop_border=True
            ),
            # Horizontal flip
            A.HorizontalFlip(p=flip_prob),
        ], seed=seed)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        """
        Apply synchronized augmentations to both image and mask.
        
        Args:
            image (PIL.Image): Original image
            mask (PIL.Image): Corresponding mask
            
        Returns:
            tuple[PIL.Image, PIL.Image]: Augmented (image, mask) pair or original if augmentation fails
        """
        for i in range(5):  # Try up to 5 times
            # Convert to numpy arrays
            image_np = np.array(image)
            mask_np = np.array(mask)

            # Apply initial augmentation
            transformed = self.transform(image=image_np, mask=mask_np)

            # Get augmented image dimensions
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
            image_height, image_width = transformed_image.shape[:2]

            # Determine dynamic crop size, ensuring it's within image dimensions
            crop_height, crop_width = self._get_dynamic_crop_size(image_height, image_width)

            # Create random crop transform
            crop_transform = A.Compose([
                A.RandomCrop(height=crop_height, width=crop_width, p=1.0),
            ], seed=self.seed + i)

            # Apply dynamic random crop
            cropped = crop_transform(image=transformed_image, mask=transformed_mask)

            # Check if mask is valid
            final_mask = cropped['mask']
            if np.any(final_mask):  # If mask contains any non-zero values
                # Convert back to PIL Image
                final_image = Image.fromarray(cropped['image'])
                return final_image, Image.fromarray(final_mask)

        # If all attempts fail, return original image and mask
        print("Augmentation failed after 5 attempts. Returning original image and mask.")
        return image, mask

    def _get_dynamic_crop_size(self, image_height, image_width):
        """
        Generate dynamic crop size based on image dimensions, ensuring valid crop size.
        
        Args:
            image_height (int): Height of the image
            image_width (int): Width of the image
        
        Returns:
            tuple[int, int]: Valid crop size (height, width)
        """
        min_height = int(self.min_crop_ratio * image_height)
        min_width = int(self.min_crop_ratio * image_width)

        crop_height = random.randint(min_height, image_height)
        crop_width = random.randint(min_width, image_width)

        return crop_height, crop_width


class MaskAugmenter:
    """
    Class to manage mask data augmentation operations.
    Supports two mutually exclusive augmentation methods: perturbation and blurring.
    """
    def __init__(
        self,
        perturb_prob: float = 0.5,
        blur_prob: float = 0.3,
        min_iou: float = 0.3,
        max_iou: float = 0.99,
        blur_factor_range: tuple = (10, 45),
        blur_threshold: tuple = (45, 128)
    ):
        """
        Initialize the data augmentation manager.
        
        Args:
            perturb_prob: Probability of mask perturbation
            blur_prob: Probability of mask blurring
            min_iou: Minimum IoU for perturbation
            max_iou: Maximum IoU for perturbation
            blur_factor_range: Range of blur factors (min, max)
            blur_threshold: Range of binarization thresholds after blurring (min, max)
        """
        self.perturb_prob = perturb_prob
        self.blur_prob = blur_prob
        self.min_iou = min_iou
        self.max_iou = max_iou
        self.blur_factor_range = blur_factor_range
        self.blur_threshold = blur_threshold
        self.mask_processor = VaeImageProcessor(
            do_normalize=False, 
            do_binarize=True,
            do_convert_grayscale=True
        )
        
        # Validate probability values
        self._validate_probs()
        
    def _validate_probs(self):
        """Validate that probability values are within valid range."""
        # Check that each probability is within [0,1]
        if not 0 <= self.perturb_prob <= 1:
            raise ValueError("perturb_prob must be between 0 and 1")
        if not 0 <= self.blur_prob <= 1:
            raise ValueError("blur_prob must be between 0 and 1")
            
        # Check that the sum of augmentation probabilities does not exceed 1
        total_prob = self.perturb_prob + self.blur_prob
        if total_prob > 1:
            raise ValueError(
                f"Sum of augmentation probabilities ({total_prob}) must be less than or equal to 1"
            )
            
    def __call__(self, mask: Image.Image) -> Image.Image:
        """
        Apply random augmentation to the input mask, using only one method per call.
        If augmentation fails, retry up to five times and return the original mask.
        
        Args:
            mask: input mask image (PIL Image)
            
        Returns:
            Augmented mask image (PIL Image)
        """
        for _ in range(5):  # Try up to 5 times
            # Generate a random value to decide which augmentation to apply
            rand_value = np.random.random()
            
            if rand_value < self.perturb_prob:
                # Apply perturbation
                augmented_mask = perturb_mask(
                    mask,
                    min_iou=self.min_iou,
                    max_iou=self.max_iou
                )
            elif rand_value < self.perturb_prob + self.blur_prob:
                # Apply blurring
                blur_factor = np.random.uniform(*self.blur_factor_range)
                threshold = np.random.uniform(*self.blur_threshold)
                augmented_mask = blur_mask(
                    self.mask_processor,
                    mask,
                    blur_factor=blur_factor,
                    threshold=threshold
                )
            else:
                # No augmentation
                augmented_mask = mask
            
            # Check if augmented mask is valid
            if np.any(np.array(augmented_mask)):  # Determine if mask contains any non-zero values
                return augmented_mask

        # If all attempts fail, return original mask
        print("Mask augmentation failed after 5 attempts. Returning original mask.")
        return mask
    
    def get_config(self) -> Dict:
        """Get current configuration."""
        return {
            "perturb_prob": self.perturb_prob,
            "blur_prob": self.blur_prob,
            "min_iou": self.min_iou,
            "max_iou": self.max_iou,
            "blur_factor_range": self.blur_factor_range,
            "blur_threshold": self.blur_threshold
        }
    
    def update_config(self, **kwargs):
        """
        Update configuration parameters.
        
        Example:
            augmentor.update_config(perturb_prob=0.3, blur_prob=0.4)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # Re-validate probability values after update
        self._validate_probs()


# From mimicbrush
# Generate a random morphological structuring element: rectangle or various shapes of ellipses
def get_random_structure(size):
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size // 2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size // 2, size))
    
# Random mask dilation
def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg, kernel, iterations=1)
    return seg

# Random mask erosion
def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg, kernel, iterations=1)
    return seg

def compute_iou(seg, gt):
    intersection = seg * gt
    union = seg + gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)

def select_max_region(mask):
    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    background = 0
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            background = row
    stats_no_bg = np.delete(stats, background, axis=0)
    max_idx = stats_no_bg[:, 4].argmax()
    max_region = np.where(labels == max_idx + 1, 1, 0)

    return max_region.astype(np.uint8)

# mask perturbation (resulting mask will be highly irregular)
# Input: gt mask: PIL.Image, min_iou: minimum IoU, max_iou: maximum IoU
# Output: seg mask: PIL.Image
def perturb_mask(gt, min_iou=0.3, max_iou=0.99):
    gt = data_utils.pil2mask(gt)
    iou_target = np.random.uniform(min_iou, max_iou)
    h, w = gt.shape
    gt = gt.astype(np.uint8)
    seg = gt.copy()
    
    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx + 1, w + 1), np.random.randint(ly + 1, h + 1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.1:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            # Dilate/erode
            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])
            
            seg = np.logical_or(seg, gt).astype(np.uint8)

        if compute_iou(seg, gt) < iou_target:
            break
    seg = select_max_region(seg.astype(np.uint8))
    seg = seg.astype(np.uint8)
    # Convert back to PIL Image object
    return data_utils.mask2pil(seg)

# Apply blurring to the mask
# threshold: smaller threshold yields a larger mask area
def blur_mask(mask_processor, mask_img, blur_factor, threshold=128):
    blurred_mask = mask_processor.blur(mask_img, blur_factor=blur_factor)
    blurred_mask = data_utils.pil2mask(blurred_mask)
    blurred_mask = (blurred_mask > threshold).astype(np.uint8)
    blurred_mask = data_utils.mask2pil(blurred_mask)
    return blurred_mask
