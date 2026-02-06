import json
import torch
import random
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import glob
import pydicom
from skimage.transform import resize
class PRGenDataset(Dataset):
    def __init__(self, data_path, mode='train', stage='vae',  image_size=(512, 1024)):
        self.mode = mode
        self.stage = stage
        self.data_path = data_path
        self.image_h, self.image_w = image_size

        self.data_list = glob.glob(os.path.join(self.data_path, "*.dcm"))
        #print('~~~~~~~~~~~~~~~~~~~~~', len(self.data_list))
        train_mode_num = int(len(self.data_list) * 0.90)
        if self.mode == 'train':
            self.data_list = self.data_list[:train_mode_num]
        elif self.mode == 'val':
            self.data_list = self.data_list[train_mode_num:]
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        try:
            data = self.data_list[index]
            #name = os.path.basename(data).replace('.dcm', '.png')
            #image = Image.open(data).convert('L')
            ds = pydicom.dcmread(data)
            img = ds.pixel_array.astype(np.float32)
            if img.ndim == 3:
                 img = img[:,:,0]   

            #print('Original img shape:', img.shape)
            mask_path = self.data_list[index].replace("ExtractedDicom", "ExtractedPNGMask").replace('.dcm', '.png')
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask, dtype=np.float32)

            # Normalize
            # -----------------------------
            img = self.percentile_normalize(img)
            mask = self.normalize_mask(mask)

            # -----------------------------
            # Resize
            # -----------------------------
            img = self.resize_image(img)
            mask = self.resize_mask(mask)

            img = torch.tensor(img, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)

            #print('img shape:', img.shape, mask.shape, data)

            image = torch.cat((img.unsqueeze(0),  mask.unsqueeze(0)), dim=0) #img.unsqueeze(0),
            #print('image shape:', image.shape)
        except Exception as e:
            #print(f"[WARN] Failed to load {self.data_list[index]}: {e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        return {"image": image}#img.unsqueeze(0)}

    def percentile_normalize(
        self,
        x: np.ndarray,
        output_min: float = -1.0,
        output_max: float = 1.0,
        clip: bool = True,
        eps: float = 1e-8,
        low_percentile: float = 10.0,
        high_percentile: float = 90.0,
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
        p_low = np.percentile(x, low_percentile)
        p_high = np.percentile(x, high_percentile)
        
        # Normalize to [0, 1] first
        x_norm = (x - p_low) / (p_high - p_low + eps)
        
        # Scale to [output_min, output_max]
        x_norm = x_norm * (output_max - output_min) + output_min
        
        if clip:
            x_norm = np.clip(x_norm, output_min, output_max)
        
        return x_norm


    def normalize_image(self, tensor):
        tensor = (tensor-np.min(tensor))/(np.max(tensor)-np.min(tensor)+1e-5)
        return tensor* 2. - 1.

    def normalize_mask(self, tensor):
        tensor = tensor/255.0
        return tensor* 2. - 1.

    def resize_image(self, img):
        """Bilinear interpolation for image"""
        return resize(img, (self.image_h, self.image_w), order=1, preserve_range=True)

    def resize_mask(self, m):
        """NEAREST interpolation for mask (avoid edge mixing)"""
        return resize(m, (self.image_h, self.image_w), order=0, preserve_range=True)



if __name__ == '__main__':
    train_MRI_dataset = PRGenDataset(data_path="xxxx/ExtractedDicom/", mode='train', stage='vae')
    val_MRI_dataset = PRGenDataset(data_path="xxxx/ExtractedDicom/", mode='val', stage='vae')
    
    train_dataset = train_MRI_dataset
    val_dataset = val_MRI_dataset
    
    train_data = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=False)
    #test_data = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)
    
    for i, data in enumerate(train_data):
        print(i)
        # print(data['image_path'])
        # print(data['image'].shape)
        # print(data['prompt'])