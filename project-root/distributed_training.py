import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, DistributedSampler
from models.fpn import FPN
from datasets.segmentation_dataset import SegmentationDataset
from utils.metrics import calculate_iou, calculate_dice
from utils.train_helpers import EarlyStopping, save_checkpoint
import torch.multiprocessing as mp
from torchvision import transforms
from torchvision.transforms import functional as TF
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Initialize DDP process group
def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

# Clean up DDP process group
def cleanup_ddp():
    dist.destroy_process_group()

# Replace SegmentationTransform class with Albumentations transforms
train_transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.Affine(shear=15, p=0.5),
    A.GaussNoise(p=0.2),
    A.CLAHE(p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.OneOf([
        A.Sharpen(p=1),
        A.Blur(blur_limit=3, p=1),
    ], p=0.5),
    A.MotionBlur(p=0.3),
    A.RandomContrast(limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    ToTensorV2()
])

val_transform = A.Compose([
    ToTensorV2()
])

# Main training function
def train(rank, world_size):
    setup_ddp(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Load training and validation datasets
    train_dataset = SegmentationDataset(
        image_dir="data/carla_lane_detection/images/train",
        label_dir="data/carla_lane_detection/masks/train",
        transform=train_transform
    )
    
    val_dataset = SegmentationDataset(
        image_dir="data/carla_lane_detection/images/val",
        label_dir="data/carla_lane_detection/masks/val",
        transform=val_transform
    )

    # Create DistributedSampler for data parallelism
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    # Dataloader with distributed sampler
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=16, sampler=val_sampler)

    # Initialize model, loss function, optimizer and scheduler
    model = FPN(num_classes=3).to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Early stopping setup
    early_stopping = EarlyStopping(patience=5, delta=0)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"Rank {rank} - Starting epoch {epoch+1}/{num_epochs}...")
        
        model.train()
        train_sampler.set_epoch(epoch)  # Shuffle data across nodes
        running_loss = 0.0

        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print progress every 10 batches
            if (i + 1) % 10 == 0:
                print(f"Rank {rank} - Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        print(f"Rank {rank} - Epoch [{epoch+1}/{num_epochs}] - Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        iou_total = 0.0
        dice_total = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device).long()
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Calculate metrics
                iou = calculate_iou(outputs, masks, num_classes=3)
                dice = calculate_dice(outputs, masks, num_classes=3)
                iou_total += iou
                dice_total += dice

        avg_val_loss = val_loss / len(val_loader)
        avg_iou = iou_total / len(val_loader)
        avg_dice = dice_total / len(val_loader)
        print(f"Rank {rank} - Validation Loss: {avg_val_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")

        # Scheduler step and early stopping
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss)

        if early_stopping.early_stop:
            print(f"Rank {rank} - Early stopping triggered at epoch {epoch+1}")
            break

        # Save checkpoint
        if rank == 0:  # Only save checkpoint from one rank
            save_checkpoint(model, optimizer, epoch, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
            print(f"Checkpoint saved for epoch {epoch+1}")
    
    cleanup_ddp()

# Main entry point to spawn multiple processes
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    world_size = 2  # Set to use 2 GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
