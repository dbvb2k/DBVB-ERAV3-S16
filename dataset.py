import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_augmentation=True):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'annotations', 'trimaps')
        
        # Define valid image extensions
        self.valid_extensions = ('.jpg', '.jpeg', '.png')
        
        # Get list of valid image files
        image_files = [f for f in os.listdir(self.images_dir) 
                      if f.lower().endswith(self.valid_extensions)]
        print(f"\nFound {len(image_files)} valid images in {self.images_dir}")
        
        # Get valid image-mask pairs
        self.valid_pairs = []
        for img_name in image_files:
            img_path = os.path.join(self.images_dir, img_name)
            mask_name = os.path.splitext(img_name)[0] + '.png'
            mask_path = os.path.join(self.masks_dir, mask_name)
            
            if os.path.exists(mask_path):
                self.valid_pairs.append((img_path, mask_path))
            else:
                print(f"Warning: Skipping {img_name} - missing mask file")
        
        print(f"Found {len(self.valid_pairs)} valid image-mask pairs")
        
        if len(self.valid_pairs) == 0:
            raise ValueError("No valid image-mask pairs found!")

        self.use_augmentation = use_augmentation
        self.img_transform = transform
        
        # Base mask transform without augmentation
        self.base_mask_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        # Mask transform with geometric augmentations (no color transforms)
        self.aug_mask_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ToTensor(),
        ]) if use_augmentation else self.base_mask_transform

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            print(f"Error loading files:\nImage: {img_path}\nMask: {mask_path}")
            raise e

        if self.img_transform:
            if self.use_augmentation:
                # Get random seed for consistent transformations
                seed = torch.randint(0, 2**32, (1,))[0].item()
                
                # Apply geometric transforms consistently
                torch.manual_seed(seed)
                image = self.img_transform(image)
                
                torch.manual_seed(seed)
                mask = self.aug_mask_transform(mask)
            else:
                # Apply basic transforms without augmentation
                image = self.img_transform(image)
                mask = self.base_mask_transform(mask)

        # Convert trimap to binary mask
        mask = (mask * 255).long()
        mask = (mask == 1).float()
        
        return image, mask 