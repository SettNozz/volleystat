import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def visualize_masks(query_dir, mask_dir, num_samples=5):
    """Visualize query images and their corresponding masks."""
    query_files = sorted([f for f in os.listdir(query_dir) if f.endswith('.jpg')])

    for i, qf in enumerate(query_files[:num_samples]):
        mask_name = qf.replace('.jpg', '_mask.png')

        query_path = os.path.join(query_dir, qf)
        mask_path = os.path.join(mask_dir, mask_name)

        query_img = Image.open(query_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.title("Image")
        plt.axis('off')
        plt.imshow(query_img)

        plt.subplot(1,2,2)
        plt.title("Mask")
        plt.axis('off')
        plt.imshow(mask_img, cmap='gray')

        plt.show()


def plot_training_losses(train_losses, val_losses, num_epochs):
    """Plot training and validation losses."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_validation_loss.png")
    plt.show()


def visualize_test_results(query_file_path, gt_mask, pred_mask_resized, name, iou, output_dir):
    """Visualize test results with original image, GT mask, and prediction."""
    orig = cv2.imread(query_file_path)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    axs[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Image")
    axs[0].axis("off")

    axs[1].imshow(gt_mask, cmap="gray")
    axs[1].set_title("GT Mask")
    axs[1].axis("off")

    axs[2].imshow(pred_mask_resized, cmap="gray")
    axs[2].set_title(f"Prediction (IoU={iou:.2f})")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_compare.png"))
    plt.close() 