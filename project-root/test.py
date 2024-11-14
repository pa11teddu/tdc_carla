import os

# Define the paths
base_dir = 'data/carla_lane_detection'
train_image_dir = os.path.join(base_dir, 'images', 'train')
val_image_dir = os.path.join(base_dir, 'images', 'val')
train_mask_dir = os.path.join(base_dir, 'masks', 'train')
val_mask_dir = os.path.join(base_dir, 'masks', 'val')

# Function to verify dataset structure without suffix
def verify_dataset(image_folder, mask_folder):
    missing_masks = []
    print(f"Checking {image_folder} for corresponding masks in {mask_folder}...")  # Debugging
    for image_file in os.listdir(image_folder):
        # Construct the expected mask filename (no suffix needed)
        mask_file = os.path.join(mask_folder, image_file)
        if not os.path.exists(mask_file):
            missing_masks.append(image_file)
    return missing_masks

# Verify training dataset
missing_train_masks = verify_dataset(train_image_dir, train_mask_dir)
if missing_train_masks:
    print("Missing training masks for the following images:", missing_train_masks)
else:
    print("All training images have corresponding masks.")

# Verify validation dataset
missing_val_masks = verify_dataset(val_image_dir, val_mask_dir)
if missing_val_masks:
    print("Missing validation masks for the following images:", missing_val_masks)
else:
    print("All validation images have corresponding masks.")
