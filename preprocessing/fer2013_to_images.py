import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# Path to FER2013 CSV file
csv_path = "data/FER2013/fer2013.csv"

# Output directory for images
output_dir = "data/FER2013/images"

# Emotion label mapping
emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# Create output folders
for emotion in emotion_map.values():
    os.makedirs(os.path.join(output_dir, emotion), exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_path)

print("Converting FER2013 CSV to images...")

# Convert each row to an image
for idx, row in tqdm(df.iterrows(), total=len(df)):
    emotion_label = emotion_map[row["emotion"]]
    pixels = np.array(row["pixels"].split(), dtype="uint8").reshape(48, 48)

    image_path = os.path.join(output_dir, emotion_label, f"{idx}.png")
    cv2.imwrite(image_path, pixels)

print("Done. Images saved successfully.")
