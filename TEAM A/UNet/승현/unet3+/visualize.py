# visualize.py
import matplotlib.pyplot as plt
import torch
import os

def show_predictions(model, dataloader, device, num_images=4):
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            masks = outputs.squeeze().cpu().numpy()
            break  # 한 batch만 시각화

    for i in range(num_images):
        plt.subplot(2, num_images, i+1)
        plt.imshow(images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5)
        plt.axis('off')
        plt.title("Input")

        plt.subplot(2, num_images, i+1+num_images)
        plt.imshow(masks[i], cmap='gray')
        plt.axis('off')
        plt.title("Predicted Mask")

    plt.tight_layout()
    plt.show()

def save_prediction_grid(model, dataloader, device, save_path="./predictions.png", num_images=4):
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            masks = outputs.squeeze().cpu().numpy()
            break

    fig = plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(2, num_images, i+1)
        plt.imshow(images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5)
        plt.axis('off')
        plt.title("Input")

        plt.subplot(2, num_images, i+1+num_images)
        plt.imshow(masks[i], cmap='gray')
        plt.axis('off')
        plt.title("Predicted Mask")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
