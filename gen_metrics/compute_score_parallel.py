import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModel, AutoModel
import utils
import os
from tqdm import tqdm
from fid_eval_torchmetrics import calculate_fid, calculate_is, calculate_ssim, calculate_psnr, calculate_lpips, crop_and_resize_bbox, crop_and_pad_to_size
import logging
import argparse


class ImagePairDataset(Dataset):
    def __init__(self, json_data, root_path, use_ref_image_to_compare=False):
        self.data = json_data
        self.root_path = root_path
        self.real_images = []
        self.fake_images = []
        self.real_images_local = []
        self.fake_images_local = []
        self.use_ref_image_to_compare = use_ref_image_to_compare
        
        logging.info("Reading images...")
        for item in self.data:
            if use_ref_image_to_compare:
                assert 'ref_image' in item, "need 'ref_image' in json file!"
                # Replace original image with reference image if comparing to reference
                item['original_image'] = item['ref_image']
            # Convert to absolute paths
            item['original_image'] = os.path.join(self.root_path, item['original_image'])
            item['generated_image'] = os.path.join(self.root_path, item['generated_image'])
            
            real_image = Image.open(item['original_image'])
            fake_image = Image.open(item['generated_image'])
            self.real_images.append(real_image)
            self.fake_images.append(fake_image)
            
            if not use_ref_image_to_compare:
                # Crop and resize around bbox region for local comparison
                self.real_images_local.append(
                    crop_and_resize_bbox(real_image, item['bbox'], scale=1.5)
                )
                self.fake_images_local.append(
                    crop_and_resize_bbox(fake_image, item['bbox'], scale=1.5)
                )
            else:
                # Use full images and pad generated image to original size for local comparison
                self.real_images_local.append(real_image.copy())
                self.fake_images_local.append(
                    crop_and_pad_to_size(fake_image, item['bbox'], size=real_image.size)
                )

            # Ensure generated and original images have the same size
            assert real_image.size == fake_image.size, (
                f"Image size mismatch: {real_image.size} vs {fake_image.size}"
            )
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        real_image = self.real_images[idx]
        fake_image = self.fake_images[idx]
        real_image_local = self.real_images_local[idx]
        fake_image_local = self.fake_images_local[idx]
        return real_image, fake_image, real_image_local, fake_image_local, item


def collate_fn(batch):
    real_images = [item[0] for item in batch]
    fake_images = [item[1] for item in batch]
    real_images_local = [item[2] for item in batch]
    fake_images_local = [item[3] for item in batch]
    items = [item[4] for item in batch]
    return real_images, fake_images, real_images_local, fake_images_local, items


class Score_computer:    
    def __init__(self, device):
        self.device = device
        
        # Load CLIP model and processor
        clip_version = "" # clip-vit-base-patch32
        self.clip_processor = AutoProcessor.from_pretrained(clip_version)
        self.clip_model = CLIPVisionModel.from_pretrained(clip_version).to(self.device)
        
        # Load DINOv2 model and processor
        dino_version = "" # dinov2-base
        self.dino_processor = AutoProcessor.from_pretrained(dino_version)
        self.dino_model = AutoModel.from_pretrained(dino_version).to(self.device)

    def compute_score(self, dataset, dataloader):
        """
        Compute all metrics for the dataset and dataloader.
        Global metrics use full images; local metrics use cropped regions.
        """
        calculate_local = bool(dataset.real_images_local and dataset.fake_images_local)
        if calculate_local:
            print("Found local images, local scores will be calculated.")
        
        results = []
        for real_image, fake_image, real_image_local, fake_image_local, items in tqdm(dataloader):
            clip_score = self.compute_clip_score(real_image, fake_image)
            dino_score = self.compute_dinov2_score(real_image, fake_image)
            ssim = calculate_ssim(real_image, fake_image)
            psnr = calculate_psnr(real_image, fake_image)
            lpips = calculate_lpips(real_image, fake_image)
            
            if calculate_local:
                local_clip_score = self.compute_clip_score(real_image_local, fake_image_local)
                local_dino_score = self.compute_dinov2_score(real_image_local, fake_image_local)
                local_ssim = calculate_ssim(real_image_local, fake_image_local)
                local_psnr = calculate_psnr(real_image_local, fake_image_local)
                local_lpips = calculate_lpips(real_image_local, fake_image_local)
            
            # Attach per-item scores
            for i in range(len(items)):
                item_score = {
                    "ssim": {"global": ssim[i], "local": local_ssim[i]},
                    "psnr": {"global": psnr[i], "local": local_psnr[i]},
                    "lpips": {"global": lpips[i], "local": local_lpips[i]},
                    "clip_score": {
                        "global": {"cls_score": clip_score[i]["cls_score"]},
                        "local": {"cls_score": local_clip_score[i]["cls_score"]}
                    },
                    "dino_score": {
                        "global": {"cls_score": dino_score[i]["cls_score"]},
                        "local": {"cls_score": local_dino_score[i]["cls_score"]}
                    },
                }
                items[i]['score'] = item_score
            results.extend(items)
        
        # Compute averaged global and local scores
        clip_cls_score = np.mean([
            float(item['score']['clip_score']['global']['cls_score']) for item in results
        ])
        dino_cls_score = np.mean([
            float(item['score']['dino_score']['global']['cls_score']) for item in results
        ])
        local_clip_cls_score = np.mean([
            float(item['score']['clip_score']['local']['cls_score']) for item in results
        ])
        local_dino_cls_score = np.mean([
            float(item['score']['dino_score']['local']['cls_score']) for item in results
        ])
        
        ssim_global = np.mean([
            float(item['score']['ssim']['global']) for item in results
        ])
        ssim_local = np.mean([
            float(item['score']['ssim']['local']) for item in results
        ])
        
        psnr_global = np.mean([
            float(item['score']['psnr']['global']) for item in results
        ])
        psnr_local = np.mean([
            float(item['score']['psnr']['local']) for item in results
        ])
        
        lpips_global = np.mean([
            float(item['score']['lpips']['global']) for item in results
        ])
        lpips_local = np.mean([
            float(item['score']['lpips']['local']) for item in results
        ])

        print("Calculating FID...")
        fid_global = calculate_fid(dataset.real_images, dataset.fake_images, device=self.device)
        if calculate_local:
            print("Calculating Local FID...")
            fid_local = calculate_fid(dataset.real_images_local, dataset.fake_images_local, device=self.device)
        
        global_score = {
            "fid": {"global": fid_global, "local": fid_local},
            "ssim": {"global": ssim_global, "local": ssim_local},
            "psnr": {"global": psnr_global, "local": psnr_local},
            "lpips": {"global": lpips_global, "local": lpips_local},
            "clip_score": {
                "global": {"cls_score": clip_cls_score},
                "local": {"cls_score": local_clip_cls_score}
            },
            "dino_score": {
                "global": {"cls_score": dino_cls_score},
                "local": {"cls_score": local_dino_cls_score}
            }
        }
        
        # Prepend global summary
        results.insert(0, global_score)
        return results
    
    def compute_item_score(self, real_image, fake_image):
        """Compute all metrics for a single image pair."""
        clip_score = self.compute_clip_score(real_image, fake_image)
        dino_score = self.compute_dinov2_score(real_image, fake_image)
        ssim = calculate_ssim(real_image, fake_image)
        psnr = calculate_psnr(real_image, fake_image)
        lpips = calculate_lpips(real_image, fake_image)
        return {
            "ssim": ssim,
            "psnr": psnr,
            "lpips": lpips,
            "clip_score": {"cls_score": clip_score[0]["cls_score"]},
            "dino_score": {"cls_score": dino_score[0]["cls_score"]}
        }

    def compute_clip_score(self, image1, image2):
        """
        Compute cosine similarity of CLIP features between two lists of images.
        Returns list of dicts with 'cls_score'.
        """
        inputs1 = self.clip_processor(images=image1, return_tensors="pt").to(self.device)
        inputs2 = self.clip_processor(images=image2, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            feats1 = self.clip_model(**inputs1).pooler_output
            feats2 = self.clip_model(**inputs2).pooler_output
            cls_score = torch.nn.functional.cosine_similarity(feats1, feats2)
        
        return [{"cls_score": cls_score[i].item()} for i in range(len(cls_score))]

    def compute_dinov2_score(self, image1, image2):
        """
        Compute cosine similarity of DINOv2 features between two lists of images.
        Returns list of dicts with 'cls_score'.
        """
        inputs1 = self.dino_processor(images=image1, return_tensors="pt").to(self.device)
        inputs2 = self.dino_processor(images=image2, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            feats1 = self.dino_model(**inputs1)[0][:, 0]
            feats2 = self.dino_model(**inputs2)[0][:, 0]
            cls_score = torch.nn.functional.cosine_similarity(feats1, feats2)
        
        return [{"cls_score": cls_score[i].item()} for i in range(len(cls_score))]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--use_ref_image_to_compare", action="store_true")

    args = parser.parse_args()
    json_path = args.json_path

    device = torch.device("cuda")
    # Create ScoreComputer instance
    score_computer = Score_computer(device)
    # Read data
    json_data = utils.read_json(json_path)
    root_path = os.path.dirname(json_path)
    dataset = ImagePairDataset(
        json_data,
        root_path=root_path,
        use_ref_image_to_compare=args.use_ref_image_to_compare
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,  # Adjust batch size as needed
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )
    
    # Compute scores
    results = score_computer.compute_score(dataset, dataloader)
    
    # Save results
    utils.save_json(results, json_path.replace(".json", "_score_try.json"))
