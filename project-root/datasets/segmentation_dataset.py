import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory with input images.
            label_dir (str): Path to the directory with corresponding masks.
            transform (callable, optional): Optional transformations to apply on images and masks.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))  # Ensure images are sorted
        self.labels = sorted(os.listdir(label_dir))  # Ensure labels match image order
        self.to_tensor = transforms.ToTensor()  # Convert images to tensors

    def __len__(self):
        # Return the number of images in the dataset
        return len(self.images)

    def __getitem__(self, idx):
        # Load an image and its corresponding mask
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        
        # Open the image and mask
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")  # Grayscale mask for segmentation
        
        # Convert images and masks to tensors
        image = self.to_tensor(image)
        label = self.to_tensor(label).squeeze(0)  # Remove the extra channel dimension for mask

        # Apply additional transformations if any
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        # Debugging: Print shapes of image and label tensors
        print(f"Loaded image shape: {image.shape}, mask shape: {label.shape}")

        return image, label
