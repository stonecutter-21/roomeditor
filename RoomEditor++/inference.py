import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as maskUtils
from diffusers import FluxFillPipeline
from utils.utils import get_bbox_from_mask, expand_bbox, pad_to_square, box2squre, crop_back, expand_image_mask

device = torch.device("cuda")
dtype = torch.bfloat16
size = (768, 768)

pipe = FluxFillPipeline.from_pretrained(
    "",
    torch_dtype=dtype
)
pipe.load_lora_weights("your path")


pipe.to(device)

def safe_imread(path):
    if not os.path.exists(path):
        print(f"[Warning] Missing image: {path}")
        return None
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def decode_rle(seg, h, w):
    # Polygon format
    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles)
        m = maskUtils.decode(rle)

    # RLE format
    elif isinstance(seg, dict):
        if isinstance(seg.get('counts'), list):  
            rle = maskUtils.frPyObjects([seg], h, w)
            m = maskUtils.decode(rle)
        else:  
            rle = dict(seg)
            if isinstance(rle['counts'], str):
                rle['counts'] = rle['counts'].encode('utf-8')
            m = maskUtils.decode(rle)

    else:
        # Fallback / Default case
        rle = maskUtils.frPyObjects(seg, h, w)
        m = maskUtils.decode(rle)

    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 0).astype(np.uint8) * 255


def process_item(item, annotations_dict, save_dir="./result", filename="1.jpg", seed=42):
    source_image_path = item["background"]["image_path"]
    ref_image_path = item["product"]["image_path"]

    src_anno = annotations_dict[item["background"]["id"]]
    ref_anno = annotations_dict[item["product"]["id"]]
    src_mask = decode_rle(
        src_anno["segmentation"],
        src_anno["segmentation"]["size"][0] if isinstance(src_anno["segmentation"], dict) else item["background"]["height"],
        src_anno["segmentation"]["size"][1] if isinstance(src_anno["segmentation"], dict) else item["background"]["width"]
    )
    ref_mask = decode_rle(
        ref_anno["segmentation"],
        ref_anno["segmentation"]["size"][0] if isinstance(ref_anno["segmentation"], dict) else item["product"]["height"],
        ref_anno["segmentation"]["size"][1] if isinstance(ref_anno["segmentation"], dict) else item["product"]["width"]
    )
    if np.sum(src_mask) == 0 or np.sum(ref_mask) == 0:
        print("Warning: src_mask or ref_mask is empty!")
        return None

    tar_image = safe_imread(source_image_path)
    ref_image = safe_imread(ref_image_path)
    if tar_image is None or ref_image is None:
        return None

    tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    tar_mask = (src_mask > 128).astype(np.uint8)
    ref_mask = (ref_mask > 128).astype(np.uint8)

    ref_box = get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask] * 3, -1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)
    y1, y2, x1, x2 = ref_box
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
    ref_mask = ref_mask[y1:y2, x1:x2]
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=1.3)
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)


    kernel = np.ones((7, 7), np.uint8)
    tar_mask = cv2.dilate(tar_mask, kernel, iterations=2)
    tar_box = get_bbox_from_mask(tar_mask)
    tar_box = expand_bbox(tar_mask, tar_box, ratio=1.2)
    tar_box_crop = expand_bbox(tar_image, tar_box, ratio=2)
    tar_box_crop = box2squre(tar_image, tar_box_crop)
    y1, y2, x1, x2 = tar_box_crop
    old_tar_image = tar_image.copy()
    tar_image = tar_image[y1:y2, x1:x2, :]
    tar_mask = tar_mask[y1:y2, x1:x2]

    H1, W1 = tar_image.shape[:2]
    tar_mask = pad_to_square(tar_mask, pad_value=0)
    tar_mask = cv2.resize(tar_mask, size)
    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)


    prompt_embeds = torch.zeros((1, 512, 4096), device=device, dtype=torch.bfloat16)
    pooled_prompt_embeds = torch.zeros((1, 768), device=device, dtype=torch.bfloat16)

    tar_image = pad_to_square(tar_image, pad_value=255)
    H2, W2 = tar_image.shape[:2]
    tar_image = cv2.resize(tar_image, size)

    diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)
    tar_mask = np.stack([tar_mask] * 3, axis=-1)
    mask_black = np.zeros_like(tar_image)
    mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)

    diptych_ref_tar = Image.fromarray(diptych_ref_tar)
    mask_diptych[mask_diptych == 1] = 255
    mask_diptych = Image.fromarray(mask_diptych)

    # Generate image
    generator = torch.Generator(device).manual_seed(seed)
    edited_image = pipe(
        image=diptych_ref_tar,
        mask_image=mask_diptych,
        height=mask_diptych.size[1],
        width=mask_diptych.size[0],
        max_sequence_length=512,
        generator=generator,
        prompt_embeds = prompt_embeds,
        pooled_prompt_embeds = pooled_prompt_embeds
    ).images[0]

    # Crop back to original size
    width, height = edited_image.size
    edited_image = edited_image.crop((width // 2, 0, width, height))
    edited_image = np.array(edited_image)
    edited_image = crop_back(edited_image, old_tar_image, np.array([H1, W1, H2, W2]), np.array(tar_box_crop))
    edited_image = Image.fromarray(edited_image)

    # Save with numbered filename
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.abspath(os.path.join(save_dir, filename))

    edited_image.save(save_path)

    return {
        "generated_image": save_path,
        "original_image": source_image_path,
        "ref_image": ref_image_path
    }


def process_json(json_file, output_json="results.json", save_dir="./result"):
    with open(json_file, 'r') as f:
        data = json.load(f)
    items = data["items"]
    annotations = {a["image_id"]: a for a in data["annotations"]}

    results = []
    counter = 1
    for item in items:
        fname = f"{counter}.jpg"
        record = process_item(item, annotations, save_dir=save_dir, filename=fname)
        if record:
            results.append(record)
            counter += 1

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_json}")



if __name__ == "__main__":
    process_json(
        "",
        output_json="",
        save_dir=""
    )