import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from dataset import data_utils

class CustomEvaluator:
    def __init__(
        self,
        checkpoint_path,
        save_path,
        dataset,
        dataloader,
        seed,
        pipeline_fn,
        model_dict,
        postprocess
    ):
        self.checkpoint_path = checkpoint_path
        self.save_path = os.path.join(checkpoint_path, save_path)

        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "images"), exist_ok=True)
        
        self.seed = seed
        self.generator = torch.Generator("cuda").manual_seed(seed)
        self.dataloader = dataloader
        self.dataset = dataset
        self.pipeline_fn = pipeline_fn
        self.model_dict = model_dict
        
        self.postprocess = postprocess
    
    # Post-process and save a batch of images
    def process_and_save_image(self, images, batch):
        curr_data_list = []
        # Save images
        for i, img in enumerate(images):
            # Process the generated image
            item = batch['raw'][i]
            img = crop_padding_and_resize(
                item['background']['height'],
                item['background']['width'],
                np.array(img)
            )

            # Save 
            # Reference and background images
            # Convert relative paths to absolute paths
            item['product']['image_path'] = os.path.join(
                self.dataset.root_path,
                item['product']['image_path']
            )
            item['background']['image_path'] = os.path.join(
                self.dataset.root_path,
                item['background']['image_path']
            )
            
            ref_image = cv2.imread(item['product']['image_path'], cv2.IMREAD_UNCHANGED)
            ref_mask = self.dataset.annotations_dict[item['product']['id']][0]['segmentation']
            ref_image = visualize_image(ref_image, ref_mask, None)

            back_image = cv2.imread(item['background']['image_path'], cv2.IMREAD_UNCHANGED)
            back_mask = self.dataset.annotations_dict[item['background']['id']][0]['segmentation']
            # Post-process: use the original background for non-mask areas
            if self.postprocess:
                mask = data_utils.seg2mask(back_mask, h=back_image.shape[0], w=back_image.shape[1])
                mask = data_utils.mask2pil(mask)
                img = postprocess_image(img, back_image, mask)
            else:
                # Convert RGB to BGR for saving
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            back_image = visualize_image(back_image, back_mask, None)
            
            # Resize ref_image height to match back_image, preserving aspect ratio
            ref_image = resize_image_with_height(ref_image, back_image.shape[0])
            # Resize generated image height to match back_image, preserving aspect ratio
            img = resize_image_with_height(img, back_image.shape[0])
            # Concatenate for visualization
            vis_image = cv2.hconcat([ref_image, back_image, img])
            # Save visualization
            vis_image_path = os.path.join("images", f"{batch['id'][i]}_all.png")  # relative path
            cv2.imwrite(os.path.join(self.save_path, vis_image_path), vis_image)
            
            img_path = os.path.join("images", f"{batch['id'][i]}_gen.png")  # relative path
            cv2.imwrite(os.path.join(self.save_path, img_path), img)
            state = {
                'id': batch['id'][i],
                'ref_image': item['product']['image_path'],
                'original_image': item['background']['image_path'],
                'generated_image': img_path,
                'visualization_image': vis_image_path,
                'bbox': self.dataset.annotations_dict[item['background']['id']][0]['bbox']
            }
            curr_data_list.append(state)
        return curr_data_list
        
    def evaluate(self, **kwargs):
        data_list = []
        for batch in tqdm(self.dataloader):
            images = self.pipeline_fn(
                batch, 
                self.model_dict,
                self.generator,
                **kwargs
            )
            curr_data_list = self.process_and_save_image(images, batch)
            data_list.extend(curr_data_list)
        with open(os.path.join(self.save_path, "1_data.json"), "w") as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        return None
    

# The post-processing here restores the image size to the original background_image size
# Note that we previously padded to a square; now we crop back to the original size
# copied from mimicbrush
def crop_padding_and_resize(ori_height, ori_width, square_image):
    scale = max(ori_height / square_image.shape[0], ori_width / square_image.shape[1])
    resized_square_image = cv2.resize(square_image, (int(square_image.shape[1] * scale), int(square_image.shape[0] * scale)))
    padding_size = max(resized_square_image.shape[0] - ori_height, resized_square_image.shape[1] - ori_width)
    if ori_height < ori_width:
        top = padding_size // 2
        bottom = resized_square_image.shape[0] - (padding_size - top)
        cropped_image = resized_square_image[top:bottom, :,:]
    else:
        left = padding_size // 2
        right = resized_square_image.shape[1] - (padding_size - left)
        cropped_image = resized_square_image[:, left:right,:]
    return cropped_image


def visualize_image(image_cv, segmentation, bbox):
    """
    Visualize the original image with COCO-format segmentation and bbox.

    :param image_cv: Original image as an OpenCV NumPy array
    :param segmentation: COCO-format segmentation, typically a nested list like [[x1, y1, x2, y2, ...]]
    :param bbox: COCO-format bbox, typically a list of four elements [x_min, y_min, width, height]
    :return: Visualized image as an OpenCV NumPy array
    """
    # Convert RGB to BGR
    image_cv = image_cv.copy()  # ensure original is not modified
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    overlay = np.zeros_like(image_cv, dtype=np.uint8)
    
    # Ensure bbox coordinates are integers
    if bbox is not None:
        x_min, y_min, box_width, box_height = map(int, bbox)
        x_max = x_min + box_width
        y_max = y_min + box_height
        # Fill bbox area
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)

    # Process segmentation data
    h, w = image_cv.shape[:2]
    if isinstance(segmentation, dict):
        # RLE format
        binary_mask = data_utils.seg2mask(segmentation, h, w)
        overlay[binary_mask > 0] = 255
    else:
        # Polygon format
        if isinstance(segmentation[0], list):
            # Multiple polygons
            for polygon in segmentation:
                points = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(overlay, [points], 255)
        else:
            # Single polygon
            points = np.array(segmentation, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(overlay, [points], 255)

    # Overlay mask onto original image
    alpha = 0.7
    output = cv2.addWeighted(image_cv, 1, overlay, alpha, 0)

    # Convert BGR back to RGB
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output


def resize_image_with_height(image, target_height):
    # Get original image dimensions
    original_height, original_width = image.shape[:2]
    
    # Calculate scale ratio
    scale_ratio = target_height / original_height
    
    # Compute new dimensions while preserving aspect ratio
    new_width = int(original_width * scale_ratio)
    new_height = target_height
    
    # Resize using cv2.resize
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


def postprocess_image(gen_image, raw_image, mask):
    # Convert to numpy arrays
    raw_image = np.array(raw_image).astype(np.uint8)
    mask = np.array(mask)
    gen_image = np.array(gen_image).astype(np.uint8)
    # Blur mask edges
    for i in range(10):
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask = mask / 255
    gen_image = gen_image[:, :, ::-1] * mask + raw_image * (1 - mask)
    gen_image = gen_image.astype(np.uint8)
    return gen_image
