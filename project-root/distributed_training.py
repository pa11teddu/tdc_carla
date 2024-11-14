import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, DistributedSampler
from models.unet import UNet
from datasets.segmentation_dataset import SegmentationDataset
from utils.metrics import calculate_iou, calculate_dice
from utils.train_helpers import EarlyStopping, save_checkpoint
import torch.multiprocessing as mp

# Initialize DDP process group
def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend="gloo",  
        init_method="env://",  # Using environment variables for torchrun
        world_size=world_size,
        rank=rank
    )

# Clean up DDP process group
def cleanup_ddp():
    dist.destroy_process_group()

# Main training function
def train(rank, world_size):
    setup_ddp(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Load training and validation datasets
    train_dataset = SegmentationDataset(
        image_dir="data/carla_lane_detection/images/train",
        label_dir="data/carla_lane_detection/masks/train",
        transform=None
    )
    
    val_dataset = SegmentationDataset(
        image_dir="data/carla_lane_detection/images/val",
        label_dir="data/carla_lane_detection/masks/val",
        transform=None
    )

    # Create DistributedSampler for data parallelism
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    # Dataloader with distributed sampler
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=16, sampler=val_sampler)

    # Initialize model, loss function, optimizer and scheduler
    model = UNet(num_classes=3).to(device)
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
    os.environ["MASTER_PORT"] = "12355"  # Change port if needed
    world_size = int(os.environ.get("WORLD_SIZE", 4))  # Number of processes (adjust as needed)
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
