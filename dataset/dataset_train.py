import random
from dataset.dataset_base import BaseDataset
import dataset.data_utils as data_utils
from dataset.data_augment import MaskAugmenter, ImageMaskAugmenter
from diffusers.image_processor import VaeImageProcessor

# Dataset for training (can also be used for testing)
class TrainDataset(BaseDataset):
    def __init__(
        self,
        data_json,
        background_mask_color="black",
        seed=42,
        # mask augmentation parameters
        perturb_prob=0,
        blur_prob=0,
        bbox_prob=0,
        # image augmentation parameters
        flip_prob=0,
        rotation_prob=0,
        max_rotation_angle=0,
        scale_prob=0,
        scale_range=0,
        min_crop_ratio=1,
        selected_ids=[]
    ):
        self.background_mask_color = background_mask_color
        assert self.background_mask_color in ['black', 'grey'], 'background_mask_color should be "black" or "grey"'
        super().__init__(data_json, selected_ids=selected_ids)
        self.seed = seed

        # initialize augmenters
        self.mask_augmentor = MaskAugmenter(
            perturb_prob=perturb_prob,
            blur_prob=blur_prob
        )
        self.image_augmentor = ImageMaskAugmenter(
            flip_prob=flip_prob,
            rotation_prob=rotation_prob,
            max_rotation_angle=max_rotation_angle,
            scale_prob=scale_prob,
            scale_range=(1 - scale_range, 1 + scale_range),
            min_crop_ratio=min_crop_ratio,
            seed=self.seed
        )
        self.bbox_ratio = bbox_prob

    def __getitem__(self, idx):
        """
        Here, we perform two main operations:
        (1) pad images to square;
        (2) apply masks on the background image.
        """
        try:
            item = super().__getitem__(idx)

            # 0. Data augmentation
            # Use bounding-box mask if selected
            if random.random() < self.bbox_ratio:
                item['background_mask'] = item['background_bbox_mask']

            # Perturb the background mask
            item['background_mask'] = self.mask_augmentor(item['background_mask'])

            # Augment the background image and its mask
            item['background_image'], item['background_mask'] = self.image_augmentor(
                item['background_image'], item['background_mask']
            )
            # Augment the reference image and its mask
            item['ref_image'], item['ref_mask'] = self.image_augmentor(
                item['ref_image'], item['ref_mask']
            )

            # 1. Pad all images to square
            ref_image = data_utils.pad_img_to_square(item['ref_image'])
            background_image = data_utils.pad_img_to_square(item['background_image'])
            ref_mask = data_utils.pad_img_to_square(item['ref_mask'], is_mask=True)
            background_mask = data_utils.pad_img_to_square(item['background_mask'], is_mask=True)

            # 2. Apply mask on reference and background images
            ref_image = data_utils.apply_mask(ref_image, ref_mask, mode='product')
            masked_background_image = data_utils.apply_mask(
                background_image,
                background_mask,
                mode='background_b' if self.background_mask_color == 'black' else 'background_g'
            )
            background_image_only_target = data_utils.apply_mask(
                background_image,
                background_mask,
                mode='product'
            )

            # 3. Simple checks
            assert ref_image.size == ref_mask.size, f"ref_image.size: {ref_image.size}, ref_mask.size: {ref_mask.size}"
            assert background_image.size == background_mask.size, f"background_image.size: {background_image.size}, background_mask.size: {background_mask.size}"
            assert masked_background_image.size == background_mask.size, f"masked_background_image.size: {masked_background_image.size}, background_mask.size: {background_mask.size}"

            return {
                'raw': item['raw'],
                'id': item['id'],
                'ref_image': ref_image,
                'ref_mask': ref_mask,
                'background_image': background_image,
                'background_mask': background_mask,
                'masked_background_image': masked_background_image,
                'background_image_only_target': background_image_only_target
            }

        except Exception as e:
            print(f"Error in TrainDataset: {e}")
            # retry with a random index on failure
            return self.__getitem__(random.randint(0, len(self) - 1))


# Process the reference image separately. Inputs are already square-padded image and mask.
def ref_image_process(ref_image, ref_mask, background_mask):
    # Form 1: Keep only the masked region
    cropped_image, cropped_mask = data_utils.crop_image_by_mask(ref_image, ref_mask)
    # Form 2: Place the reference image into the background at the mask location
    ref_image_in_back, ref_mask_in_back = data_utils.resize_to_fit_mask(
        cropped_image, cropped_mask, background_mask
    )

    # Then pad both Form 1 and Form 2 results to square
    cropped_image = data_utils.pad_img_to_square(cropped_image)
    cropped_mask = data_utils.pad_img_to_square(cropped_mask, is_mask=True)
    ref_image_in_back = data_utils.pad_img_to_square(ref_image_in_back)
    ref_mask_in_back = data_utils.pad_img_to_square(ref_mask_in_back, is_mask=True)

    return cropped_image, cropped_mask, ref_image_in_back, ref_mask_in_back


# Some settings: define constants here due to their fixed nature
height, width = 512, 512
vae_config_block_out_channels = [128, 256, 512, 512]
# Compute image scale factor: number of decoder layers = len(block_out_channels)-1, each with stride=2
vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)  # 8

# Image and mask processors
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
mask_processor = VaeImageProcessor(
    vae_scale_factor=vae_scale_factor,
    do_normalize=False,
    do_binarize=True,
    do_convert_grayscale=True
)

# Place data preprocessing in collate_fn for faster training
def _collate_fn(batch):
    """
    Collate function for DataLoader.

    Args:
        batch: list of samples

    Returns:
        dict: processed image pairs and metadata tensors.
    """
    ref_images = [sample['ref_image'] for sample in batch]
    ref_masks = [sample['ref_mask'] for sample in batch]


    background_images = [sample['background_image'] for sample in batch]
    masked_background_images = [sample['masked_background_image'] for sample in batch]
    background_masks = [sample['background_mask'] for sample in batch]
    background_image_only_target = [sample['background_image_only_target'] for sample in batch]

    # clip_images left for image encoder processing
    clip_images = ref_images

    ref_images = image_processor.preprocess(ref_images, height=height, width=width)
    background_images = image_processor.preprocess(background_images, height=height, width=width)
    masked_background_images = image_processor.preprocess(masked_background_images, height=height, width=width)
    background_masks = mask_processor.preprocess(background_masks, height=height, width=width)
    background_image_only_target = image_processor.preprocess(
        background_image_only_target, height=height, width=width
    )

    return {
        'id': [sample['id'] for sample in batch],
        'raw': [sample['raw'] for sample in batch],
        'clip_images': clip_images,
        'ref_images': ref_images,
        'ref_masks': ref_masks,
        'background_images': background_images,
        'masked_background_images': masked_background_images,
        'background_masks': background_masks,
        'background_image_only_target': background_image_only_target
    }
