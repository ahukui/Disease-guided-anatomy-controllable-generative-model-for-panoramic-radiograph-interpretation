import json
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import pydicom
from skimage.transform import resize
from typing import Union, Tuple, Optional


class PRGenDataset(Dataset):
    def __init__(
        self, 
        data_path: str, 
        mode: str = 'train', 
        from_json: bool = True,  
        image_size: Tuple[int, int] = (512, 1024),
        low_percentile: float = 10.0,
        high_percentile: float = 90.0,
        train_ratio: float = 0.99,
    ):
        """
        Args:
            data_path: path to folder with PNGs OR JSON file (if from_json=True)
            mode: 'train' or 'test'
            from_json: if True, expects a JSON file with {"images": "...", "content": "..."}
            image_size: (height, width) tuple
            low_percentile: lower percentile for normalization (default: 10)
            high_percentile: upper percentile for normalization (default: 90)
            train_ratio: ratio of data for training (default: 0.99)
        """
        self.mode = mode
        self.from_json = from_json
        self.image_h, self.image_w = image_size
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile

        if from_json:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data_list = json.load(f)
        else:
            # fall back to PNG folder mode
            from glob import glob
            pngs = glob(os.path.join(data_path, "*.png"))
            self.data_list = [{"images": p, "content": ""} for p in pngs]

        print(f"[INFO] Loaded {len(self.data_list)} samples")

        # Split train/test
        train_mode_num = int(len(self.data_list) * train_ratio)
        if self.mode == 'train':
            self.data_list = self.data_list[:train_mode_num]
        elif self.mode == 'test':
            self.data_list = self.data_list[train_mode_num:]
        
        print(f"[INFO] {self.mode} set: {len(self.data_list)} samples")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> dict:
        item = self.data_list[index]
        img_path = item["images"].replace("OriExtractedPNG", "ExtractedDicom").replace('.png', '.dcm')
        mask_path = item["images"].replace("OriExtractedPNG", "ExtractedPNGMask")
        skeleton_path = item["images"].replace("OriExtractedPNG", "centerline_results")
        text = item["content"]

        try:
            # Load DICOM image
            ds = pydicom.dcmread(img_path)
            image = ds.pixel_array.astype(np.float32)
            if image.ndim == 3:
                image = image[:, :, 0]
            
            # Load mask
            with Image.open(mask_path) as mask:
                mask = mask.convert('L')
            #mask = Image.open(mask_path).convert("L")
            mask = np.array(mask, dtype=np.float32)
            
            # Load skeleton
            skeleton = Image.open(skeleton_path).convert("L")
            skeleton = np.array(skeleton, dtype=np.float32)
            
        except Exception as e:
            #print(f"[WARN] Failed to load {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Normalize
        img = self.percentile_normalize(image)
        mask = self.normalize_mask(mask)
        skeleton = self.normalize_mask(skeleton)

        # Resize
        img = self.resize_image(img)
        mask = self.resize_mask(mask)
        skeleton = self.resize_mask(skeleton)

        # Convert to tensors
        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        skeleton = torch.tensor(skeleton, dtype=torch.float32)
        image = torch.cat((img.unsqueeze(0), mask.unsqueeze(0)), dim=0) #img.unsqueeze(0)

        final_name = os.path.basename(img_path)

        return {
            "image": img.unsqueeze(0),           # (1, H, W)
            "prompt": text,
            "skeleton": skeleton.unsqueeze(0),    # (1, H, W)
            "mask": mask.unsqueeze(0),            # (1, H, W)
            "image_path": final_name,
        }

    def percentile_normalize(
        self,
        x: np.ndarray,
        output_min: float = -1.0,
        output_max: float = 1.0,
        clip: bool = True,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Normalize data using percentiles to [-1, 1] range.
        
        Args:
            x: Input numpy array
            output_min: Minimum output value (default: -1)
            output_max: Maximum output value (default: 1)
            clip: Whether to clip values outside the output range
            eps: Small value to avoid division by zero
        
        Returns:
            Normalized array in [output_min, output_max] range
        """
        p_low = np.percentile(x, self.low_percentile)
        p_high = np.percentile(x, self.high_percentile)
        
        # Normalize to [0, 1] first
        x_norm = (x - p_low) / (p_high - p_low + eps)
        
        # Scale to [output_min, output_max]
        x_norm = x_norm * (output_max - output_min) + output_min
        
        if clip:
            x_norm = np.clip(x_norm, output_min, output_max)
        
        return x_norm

    def normalize_mask(self, tensor: np.ndarray) -> np.ndarray:
        """Normalize mask to [0, 1] range."""
        tensor = tensor/255.0        
        return tensor* 2. - 1.

    def normalize_image_minmax(self, tensor: np.ndarray) -> np.ndarray:
        """Alternative: min-max normalization to [-1, 1]."""
        tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor) + 1e-5)
        return tensor * 2.0 - 1.0

    def resize_image(self, img: np.ndarray) -> np.ndarray:
        """Bilinear interpolation for image."""
        return resize(img, (self.image_h, self.image_w), order=1, preserve_range=True)

    def resize_mask(self, m: np.ndarray) -> np.ndarray:
        """NEAREST interpolation for mask (avoid edge mixing)."""
        return resize(m, (self.image_h, self.image_w), order=0, preserve_range=True)


class PRGenDatasetWithAugmentation(PRGenDataset):
    """Extended dataset with data augmentation for training."""
    
    def __init__(
        self,
        data_path: str,
        mode: str = 'train',
        from_json: bool = True,
        image_size: Tuple[int, int] = (512, 1024),
        low_percentile: float = 10.0,
        high_percentile: float = 90.0,
        train_ratio: float = 0.99,
        # Augmentation parameters
        enable_augmentation: bool = True,
        horizontal_flip_prob: float = 0.5,
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
    ):
        super().__init__(
            data_path=data_path,
            mode=mode,
            from_json=from_json,
            image_size=image_size,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            train_ratio=train_ratio,
        )
        
        self.enable_augmentation = enable_augmentation and (mode == 'train')
        self.horizontal_flip_prob = horizontal_flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __getitem__(self, index: int) -> dict:
        item = self.data_list[index]
        img_path = item["images"].replace("OriExtractedPNG", "ExtractedDicom").replace('.png', '.dcm')
        mask_path = item["images"].replace("OriExtractedPNG", "ExtractedPNGMask")
        skeleton_path = item["images"].replace("OriExtractedPNG", "centerline_results")
        text = item["content"]

        try:
            # Load DICOM image
            ds = pydicom.dcmread(img_path)
            image = ds.pixel_array.astype(np.float32)
            if image.ndim == 3:
                image = image[:, :, 0]
            
            # Load mask
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask, dtype=np.float32)
            
            # Load skeleton
            skeleton = Image.open(skeleton_path).convert("L")
            skeleton = np.array(skeleton, dtype=np.float32)
            
        except Exception as e:
            #print(f"[WARN] Failed to load {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Apply augmentation BEFORE normalization (on raw pixel values)
        if self.enable_augmentation:
            image, mask, skeleton = self.apply_augmentation(image, mask, skeleton)

        # Normalize
        img = self.percentile_normalize(image)
        mask = self.normalize_mask(mask)
        skeleton = self.normalize_mask(skeleton)

        # Resize
        img = self.resize_image(img)
        mask = self.resize_mask(mask)
        skeleton = self.resize_mask(skeleton)

        # Convert to tensors
        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        skeleton = torch.tensor(skeleton, dtype=torch.float32)

        final_name = os.path.basename(img_path)

        return {
            "image": img.unsqueeze(0),
            "prompt": text,
            "skeleton": skeleton.unsqueeze(0),
            "mask": mask.unsqueeze(0),
            "image_path": final_name,
        }

    def apply_augmentation(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        skeleton: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply data augmentation to image, mask, and skeleton."""
        
        # Horizontal flip
        if random.random() < self.horizontal_flip_prob:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
            skeleton = np.fliplr(skeleton).copy()
        
        # Brightness adjustment (only on image)
        brightness_factor = random.uniform(*self.brightness_range)
        image = image * brightness_factor
        
        # Contrast adjustment (only on image)
        contrast_factor = random.uniform(*self.contrast_range)
        mean_val = np.mean(image)
        image = (image - mean_val) * contrast_factor + mean_val
        
        return image, mask, skeleton


# =============================================================================
# Utility functions
# =============================================================================

def get_dataloaders(
    data_path: str,
    train_batch_size: int = 8,
    val_batch_size: int = 1,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 1024),
    low_percentile: float = 10.0,
    high_percentile: float = 90.0,
    enable_augmentation: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    DatasetClass = PRGenDatasetWithAugmentation if enable_augmentation else PRGenDataset
    
    train_dataset = DatasetClass(
        data_path=data_path,
        mode='train',
        image_size=image_size,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        enable_augmentation=enable_augmentation if enable_augmentation else False,
    )
    
    val_dataset = PRGenDataset(
        data_path=data_path,
        mode='test',
        image_size=image_size,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    return train_loader, val_loader


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PRGenDataset Test")
    print("=" * 60)
    
    # JSON mode test
    train_dataset = PRGenDataset(
        data_path="./SixDieasesPatientInf.json",
        mode='train',
        low_percentile=10.0,
        high_percentile=90.0,
    )
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, 
        num_workers=4,  # Set to 0 for debugging
        shuffle=False
    )
    
    for idx, batch in enumerate(train_loader):
        print(f"\nSample #{idx}")
        print("-" * 40)

        # Image
        img = batch['image']
        print(f"image:")
        print(f"  shape  : {img.shape}")
        print(f"  dtype  : {img.dtype}")
        print(f"  min/max: [{img.min().item():.3f}, {img.max().item():.3f}]")

        # Mask
        mask = batch['mask']
        print(f"mask:")
        print(f"  shape  : {mask.shape}")
        print(f"  dtype  : {mask.dtype}")
        print(f"  unique : {torch.unique(mask)}")

        # Skeleton
        skel = batch['skeleton']
        print(f"skeleton:")
        print(f"  shape  : {skel.shape}")
        print(f"  dtype  : {skel.dtype}")
        print(f"  min/max: [{skel.min().item():.3f}, {skel.max().item():.3f}]")

        # Metadata
        print(f"prompt     : {batch['prompt'][0]}")
        print(f"image_path : {batch['image_path'][0]}")

    print("\n" + "=" * 80)
    print("Dataset inspection finished successfully.")
    print("=" * 80)