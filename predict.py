import torch
import cv2
import numpy as np
from datetime import datetime, timedelta
from data_extraction import extract_frames
from model import SolarRadiationPredictor
from torchvision import transforms
import matplotlib.pyplot as plt

# Load the pre-trained model
def load_model(model_path, device):
    model = SolarRadiationPredictor()
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    model.to(device)
    model.eval()
    return model

# Preprocess video frames
def preprocess_frame(frame):
    # Convert frame to a PyTorch tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor (C, H, W)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    return transform(frame).unsqueeze(0)  # Add batch dimension

# Display a frame
def display_frame(frame, frame_index):
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_HSV2RGB))  # Convert HSV to RGB for display
    plt.title(f"Frame Index: {frame_index}")
    plt.axis("off")
    plt.show()

# Predict solar radiation 5 minutes into the future
def predict_future_radiation(video_path, sensor_reading, model_path, frame_index, device):
    # Extract frames from the video
    frames = extract_frames(video_path)
    print(f"Extracted {len(frames)} frames from the video.")

    # Validate frame index
    if frame_index < 0 or frame_index >= len(frames):
        raise ValueError(f"Frame index {frame_index} is out of range. Valid range: 0 to {len(frames) - 1}")

    # Load the model
    model = load_model(model_path, device)

    # Get the selected frame
    selected_frame = frames[frame_index]

    # Display the selected frame
    display_frame(selected_frame, frame_index)

    # Preprocess the frame
    processed_frame = preprocess_frame(selected_frame).to(device)

    # Preprocess the sensor data
    sensor_tensor = torch.tensor([sensor_reading], dtype=torch.float32).to(device)

    # Predict the solar radiation 5 minutes into the future
    with torch.no_grad():
        prediction = model(processed_frame, sensor_tensor)

    return prediction.item()

# Example usage
if __name__ == "__main__":
    # Paths and device
    video_path = r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\LastYear.mp4"
    model_path = r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\models\trained_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example sensor reading
    current_sensor_reading = 223.0

    # Choose a frame index (e.g., 50th frame)
    frame_index = 500

    # Predict future solar radiation
    predicted_radiation = predict_future_radiation(video_path, current_sensor_reading, model_path, frame_index, device)
    print(f"Predicted solar radiation in 5 minutes: {predicted_radiation:.2f}")