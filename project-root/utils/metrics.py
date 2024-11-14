import torch

def calculate_iou(preds, masks, num_classes):
    """
    Calculate Intersection over Union (IoU) score.
    
    Args:
        preds (Tensor): Model predictions (logits) with shape [batch_size, num_classes, H, W].
        masks (Tensor): Ground truth masks with shape [batch_size, H, W].
        num_classes (int): Number of segmentation classes.
        
    Returns:
        float: Average IoU score across all classes.
    """
    ious = []
    preds = torch.argmax(preds, dim=1)  # Convert logits to predicted classes
    
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        mask_cls = (masks == cls).float()
        
        intersection = torch.sum(pred_cls * mask_cls)
        union = torch.sum(pred_cls) + torch.sum(mask_cls) - intersection
        
        if union == 0:
            ious.append(1.0)  # Perfect IoU if there's no union
        else:
            ious.append((intersection / union).item())
    
    return sum(ious) / len(ious)  # Average IoU across classes

def calculate_dice(preds, masks, num_classes):
    """
    Calculate Dice coefficient.
    
    Args:
        preds (Tensor): Model predictions (logits) with shape [batch_size, num_classes, H, W].
        masks (Tensor): Ground truth masks with shape [batch_size, H, W].
        num_classes (int): Number of segmentation classes.
        
    Returns:
        float: Average Dice coefficient across all classes.
    """
    dices = []
    preds = torch.argmax(preds, dim=1)  # Convert logits to predicted classes
    
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        mask_cls = (masks == cls).float()
        
        intersection = torch.sum(pred_cls * mask_cls)
        dice = (2 * intersection) / (torch.sum(pred_cls) + torch.sum(mask_cls))
        
        if torch.sum(pred_cls) + torch.sum(mask_cls) == 0:
            dices.append(1.0)  # Perfect Dice if there's no union
        else:
            dices.append(dice.item())
    
    return sum(dices) / len(dices)  # Average Dice across classes
