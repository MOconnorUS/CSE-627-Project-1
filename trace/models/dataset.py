import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class SkeletonizationDataset(Dataset):
    def __init__(self, image_dir, skele_dir, transform=None):
        self.image_dir = image_dir
        self.skele_dir = skele_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        skele_path = os.path.join(self.skele_dir, self.images[index].replace("image", "target"))
        image = np.array(Image.open(img_path).convert("L"))
        skele = np.array(Image.open(skele_path).convert("L"), dtype=np.float32)
        skele[skele == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, skele=skele)
            image = augmentations['image']
            skele = augmentations['skele']

        return image, skele
