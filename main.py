from csv_reader import load_and_combine_csvs
from data_extraction import extract_frames
from Custom_dataset import SolarRadiationImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
import cv2

# Define paths (adjust as needed)
csv_directory = r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\solar_radiation data"
video_path = r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\3928.mp4"

# Load CSV data (the CSV reader code already parses the "Timestamp" column)
combined_df = load_and_combine_csvs(csv_directory)
print("Combined DataFrame loaded with shape:", combined_df.shape)

# Extract video frames using your image extraction module
video_frames = extract_frames(video_path)
print(f"Extracted {len(video_frames)} video frames.")

# Determine the frame rate from the video file (using OpenCV)
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print("Detected frame rate:", frame_rate)

# Define the video start time manually (adjust based on your data).
# For example, if the video started at "2025-01-01 00:00:00":
video_start_time = datetime(2025, 1, 1, 0, 59, 0)

# (Optional) Define an image transform if required
image_transform = None  # or use torchvision.transforms as needed

# Create the dataset instance using time-based alignment
dataset = SolarRadiationImageDataset(
    dataframe=combined_df,
    video_frames=video_frames,
    video_start_time=video_start_time,
    frame_rate=frame_rate,
    image_transform=image_transform
)

# Wrap the dataset with a DataLoader for batching during training
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # use shuffle=False for time-based sequence if necessary

# Iterate through a few batches for testing
for batch_idx, batch in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    print("  Timestamps:", batch["timestamp"])
    print("  Solar Radiation Sensor:", batch["solar_radiation"])
    print("  Image batch shape:", batch["image"].shape)  # Expected (batch_size, 3, H, W)
    if batch_idx == 1:  # Stop after a couple of batches for testing
        break
