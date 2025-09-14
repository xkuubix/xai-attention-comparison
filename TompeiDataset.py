from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
from pydicom import dcmread
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as T
import logging
import nibabel as nib
logger = logging.getLogger(__name__)

class TompeiDataset(Dataset):
    def __init__(self, df, transform=None, config=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image metadata.
            image_dir (str): Directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.class_name = df['ClassificationMapped']
        self.transform = transform
        logger.info(f"Dataset initialized with {len(self.df)} samples.")
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.df.iloc[idx]['ImagePath'])
        image = dcmread(img_name)
        if image.BitsStored == 8: # ToTensor handles uint8
            image = image.pixel_array
        elif image.BitsStored == 12 or image.BitsStored == 16:
            bits = image.BitsStored
            max_val = float(2**bits - 1)
            image = (image.pixel_array / max_val).astype(np.float32)
        else:
            raise ValueError(f"Unsupported BitsStored: {image.BitsStored}")
        
        if image.ndim == 2:
            image = np.stack((image,)*3, axis=-1)

        mask_path = self.df.iloc[idx]['MaskPath']
        if os.path.exists(f"{mask_path}"):
            if mask_path.endswith('.nii.gz'):
                mask_nii = nib.load(mask_path)
                mask = mask_nii.get_fdata()
                mask = (mask > 0).astype(np.uint8)  # lesion mask to binary 
                mask = Image.fromarray(mask)
            else:
                mask = Image.open(mask_path)
        else:
            mask = torch.zeros(image.shape[:2], dtype=torch.uint8)
       
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.array(mask, dtype=np.uint8))

        label_str = self.df.iloc[idx]['ClassificationMapped']
        if self.config:
            if self.config['training_plan']['criterion'].lower() == 'bce':
                label = 0. if label_str == 'Negative' else 1.
                label = torch.tensor(label, dtype=torch.float32)
            elif self.config['training_plan']['criterion'].lower() == 'ce':
                label = 0 if label_str == 'Negative' else 1
                label = torch.tensor(label, dtype=torch.long)

        metadata = self.df.iloc[idx].to_dict()
        if metadata['MaskPath'] == None:
            metadata['MaskPath'] = ' '
        if self.transform:
            image = self.transform(image)

        if self.df.iloc[idx]['LeftRight'] == 'R':
            image = T.RandomHorizontalFlip(p=1.0)(image)
            mask = TF.hflip(mask)

        mask = np.array(mask, dtype=np.bool)
        # print(f"[DEBUG] idx={idx}, image={type(image)}, mask={type(mask)}, label={type(label)}, metadata={type(metadata)}")


        return {
            "image": image,
            "target": {
                "mask": mask,
                "label": label,
            },
            "metadata": metadata
        }
