import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import dataset.data_utils as data_utils
from pycocotools import mask as mask_utils


def bbox2mask(bbox, h, w):
    bbox = [int(x) for x in bbox]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    mask[y1:y1+y2, x1:x1+x2] = 1
    return mask

# Basic dataset: returns PIL images without modification
# Basic checks: ensure image and corresponding mask dimensions match
class BaseDataset(Dataset):
    def __init__(self, data_json, selected_ids=[]):
        with open(data_json, 'r') as f:
            data = json.load(f)
        if len(selected_ids) > 0:
            # If selected_ids is provided, keep only those items
            data_dict = {item['id']: item for item in data['items']}
            assert len(data_dict) == len(data['items']), "错误：数据集中id重复！！"
            print(f"Warning: 仅保留{len(selected_ids)}个数据")
            data['items'] = [data_dict[id] for id in selected_ids]
        self.data = data['items']
        # Dataset structure: the parent directory of data_json is the root path
        self.root_path = '/'.join(data_json.split('/')[:-1])
        
        annotations = data['annotations']
        self.annotations_dict = {}
        for ann in annotations:
            if ann['image_id'] not in self.annotations_dict:
                self.annotations_dict[ann['image_id']] = []
            self.annotations_dict[ann['image_id']].append(ann)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Reference image
        ref_image_path = item['product']['image_path']
        ref_image = Image.open(os.path.join(self.root_path, ref_image_path))
        # Ensure ref_image is in RGB mode, not single-channel
        if ref_image.mode != 'RGB':
            # print(f"Warning: {ref_image_path} is in {ref_image.mode} mode, convert to RGB")
            ref_image = ref_image.convert('RGB')
            
        # Reference image mask
        ref_mask = self.annotations_dict[item['product']['id']][0]['segmentation']
        ref_mask = data_utils.seg2mask(ref_mask, ref_image.size[1], ref_image.size[0])
        ref_mask = data_utils.mask2pil(ref_mask)
        
        # Background image
        background_image_path = item['background']['image_path']
        background_image = Image.open(os.path.join(self.root_path, background_image_path))
        
        # Background image mask
        background_mask = self.annotations_dict[item['background']['id']][0]['segmentation']
        background_mask = data_utils.seg2mask(
            background_mask,
            background_image.size[1],
            background_image.size[0]
        )
        background_mask = data_utils.mask2pil(background_mask)
        # Background image bounding-box mask
        background_bbox_mask = self.annotations_dict[item['background']['id']][0]['bbox']
        background_bbox_mask = bbox2mask(
            background_bbox_mask,
            background_image.size[1],
            background_image.size[0]
        )
        background_bbox_mask = data_utils.mask2pil(background_bbox_mask)

        # Simple validity checks
        # Dimension check: image and mask sizes must match
        assert ref_image.size == ref_mask.size, f"ref_image size {ref_image.size} != ref_mask size {ref_mask.size}"
        assert background_image.size == background_mask.size, f"background_image size {background_image.size} != background_mask size {background_mask.size}"
        # Mode check: images must be in RGB
        assert ref_image.mode == 'RGB', f"ref_image mode {ref_image.mode} != RGB"
        assert background_image.mode == 'RGB', f"background_image mode {background_image.mode} != RGB"
        assert ref_mask.mode == 'RGB', f"ref_mask mode {ref_mask.mode} != RGB"
        assert background_mask.mode == 'RGB', f"background_mask mode {background_mask.mode} != RGB"
        
        return {
            'raw': item,
            'id': item['id'],
            'ref_image': ref_image,
            'ref_mask': ref_mask,
            'background_image': background_image,
            'background_mask': background_mask,
            'background_bbox_mask': background_bbox_mask
        }
