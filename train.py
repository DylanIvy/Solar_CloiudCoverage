# train.py

from csv_reader import load_and_combine_csvs
from data_extraction import extract_frames
from Custom_dataset import SolarRadiationImageDataset
from model import SolarRadiationPredictor
from torch.utils.data import DataLoader
from datetime import datetime
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

# ---------- 1. Data Loading and Preparation ----------

csv_directory = r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\solar_radiation data"
test_directory = r"C:\Users\jldag\OneDrive\Desktop\test_directory"
video_path = r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\output.mp4"
test_path = r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\1-1-25.mp4"

# Load and combine CSVs
combined_df = load_and_combine_csvs(csv_directory)
test_df = load_and_combine_csvs(test_directory)

# Extract video frames and compute cloud coverage
video_frames, cloud_coverage_data = extract_frames(video_path)
test_frames, test_cloud_coverage = extract_frames(test_path)

# Get the video frame rate
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
cap.release()

print(f"Extracted {len(video_frames)} training frames.")
print(f"Extracted {len(test_frames)} test frames.")

# Set the video start time.
video_start_time = datetime(2025, 1, 1, 0, 0, 0)

# Set the forecast offset (5 minutes ahead prediction)
forecast_offset = 5

# Create dataset objects
train_dataset = SolarRadiationImageDataset(
    dataframe=combined_df,
    video_frames=video_frames,
    video_start_time=video_start_time,
    frame_rate=frame_rate,
    forecast_offset=forecast_offset,
    image_transform=None,
    cloud_coverage_data=cloud_coverage_data  # Pass cloud coverage
)

val_dataset = SolarRadiationImageDataset(
    dataframe=test_df,
    video_frames=test_frames,
    video_start_time=video_start_time,
    frame_rate=frame_rate,
    forecast_offset=forecast_offset,
    image_transform=None,
    cloud_coverage_data=test_cloud_coverage  # Pass cloud coverage
)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ---------- 2. Model, Loss, Optimizer ----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SolarRadiationPredictor().to(device)

# Loss function: Mean Squared Error (MSE) for multi-output regression
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model_save_dir = r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\models"
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, "trained_model.pth")
checkpoint_save_path = os.path.join(model_save_dir, "model_checkpoints.pth")

# ---------- 3. Training Loop with Cloud Coverage Evaluation ----------

num_epochs = 50
train_losses, val_losses = [], []
train_mae_clouds, val_mae_clouds = [], []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    total_cloud_error = 0.0

    for batch in train_dataloader:
        images = batch["image"].to(device)  # (batch_size, 3, H, W)
        sensor_data = batch["current_sensor"].float().to(device)  # (batch_size,)
        targets = batch["target"].to(device)  # (batch_size, 2) → [Solar Radiation, Cloud Coverage]

        optimizer.zero_grad()
        outputs = model(images, sensor_data)  # (batch_size, 2)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * images.size(0)

        # Calculate Cloud Coverage MAE
        cloud_pred = outputs[:, 1]  # Predicted cloud coverage
        cloud_actual = targets[:, 1]  # Actual cloud coverage
        total_cloud_error += torch.abs(cloud_pred - cloud_actual).sum().item()

    train_loss = running_train_loss / len(train_dataset)
    train_mae = total_cloud_error / len(train_dataset)
    train_losses.append(train_loss)
    train_mae_clouds.append(train_mae)

    # ---------- Validation Phase ----------
    model.eval()
    running_val_loss = 0.0
    total_val_cloud_error = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            images = batch["image"].to(device)
            sensor_data = batch["current_sensor"].float().to(device)
            targets = batch["target"].to(device)

            outputs = model(images, sensor_data)

            loss = criterion(outputs, targets)
            running_val_loss += loss.item() * images.size(0)

            # Cloud Coverage MAE
            cloud_pred = outputs[:, 1]
            cloud_actual = targets[:, 1]
            total_val_cloud_error += torch.abs(cloud_pred - cloud_actual).sum().item()

    val_loss = running_val_loss / len(val_dataset)
    val_mae = total_val_cloud_error / len(val_dataset)
    val_losses.append(val_loss)
    val_mae_clouds.append(val_mae)

    print(f"Epoch {epoch + 1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"  Train Cloud Coverage MAE: {train_mae:.2f}%, Val Cloud Coverage MAE: {val_mae:.2f}%")

    # Save model and checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }
        torch.save(checkpoint, checkpoint_save_path)
        torch.save(model.state_dict(), model_save_path)

print("Training completed.")

# ---------- 4. Plot Cloud Coverage Prediction Performance ----------
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_mae_clouds, label="Train Cloud Coverage MAE")
plt.plot(range(1, num_epochs + 1), val_mae_clouds, label="Val Cloud Coverage MAE")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error (Cloud Coverage %)")
plt.title("Cloud Coverage Prediction Accuracy Over Epochs")
plt.legend()
plt.grid()
plt.show()
