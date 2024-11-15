import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models.fpn import FPN
from datasets.segmentation_dataset import SegmentationDataset
from utils.metrics import calculate_iou, calculate_dice
from utils.train_helpers import EarlyStopping, save_checkpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 3  # Adjust based on your dataset

# Initialize dataset and dataloaders
print("Loading datasets...")

# Define training augmentations
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

# Define validation transforms (only normalization and conversion to tensor)
val_transform = A.Compose([
    ToTensorV2()
])

train_dataset = SegmentationDataset(
    image_dir="data/carla_lane_detection/images/train",
    label_dir="data/carla_lane_detection/masks/train",
    transform=train_transform)

val_dataset = SegmentationDataset(
    image_dir="data/carla_lane_detection/images/val",
    label_dir="data/carla_lane_detection/masks/val",
    transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
print("Datasets loaded successfully.")

# Initialize model, loss function, optimizer, and scheduler
print("Initializing model...")
model = FPN(num_classes=num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
early_stopping = EarlyStopping(patience=5, delta=0)
print("Model initialized successfully.")

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}/{num_epochs}...")
    model.train()
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
            print(f"Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Average Training Loss: {avg_train_loss:.4f}")

    # Validation
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
            
            iou = calculate_iou(outputs, masks, num_classes)
            dice = calculate_dice(outputs, masks, num_classes)
            iou_total += iou
            dice_total += dice

    avg_val_loss = val_loss / len(val_loader)
    avg_iou = iou_total / len(val_loader)
    avg_dice = dice_total / len(val_loader)
    print(f"Validation Results - Loss: {avg_val_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")

    # Step scheduler based on validation loss
    scheduler.step(avg_val_loss)
    early_stopping(avg_val_loss)

    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

    # Save checkpoint if validation loss improved
    save_checkpoint(model, optimizer, epoch, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
    print(f"Checkpoint saved for epoch {epoch+1}")
