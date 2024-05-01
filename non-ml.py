import numpy as np
import pandas as pd
import cv2
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import pickle

falls_dir = "data/fall/"
normal_dir = "data/normal/"
test_dir = "data/test/"

person_low = 20
person_high = 100

model = None

# # Load the model from disk
# with open(filename, 'rb') as file:
#     model = pickle.load(file)

def load_vid(csv_file_path):
    df = pd.read_csv(csv_file_path, header=None, skiprows=1)
    df.drop(columns=[0], inplace=True)
    data = df.values
    data = data.astype(np.float32).reshape((-1, 24, 32))
    return data

def find_median_point(frame, threshold):
    y_indices, x_indices = np.where(frame > threshold)  # Threshold to focus on warmer parts
    if len(y_indices) == 0 or len(x_indices) == 0:  # If no points above the threshold
        return None
    median_x = np.median(x_indices)
    median_y = np.median(y_indices)
    return median_x, median_y

def track_median_points(video, threshold):
    median_points = [find_median_point(frame, threshold) for frame in video]
    median_points = [point for point in median_points if point is not None]  # Remove None values
    return median_points

def extract_features_from_median_series(median_series):
    if not median_series:
        return np.zeros(12)  # Return a zero-filled array if series is empty

    x_coords, y_coords = zip(*median_series)
    x_velocities = np.diff(x_coords)  # Change in x
    y_velocities = np.diff(y_coords)  # Change in y

    features = [
        np.mean(x_coords), np.std(x_coords),
        np.mean(y_coords), np.std(y_coords),
        np.mean(x_velocities), np.std(x_velocities), np.min(x_velocities), np.max(x_velocities),
        np.mean(y_velocities), np.std(y_velocities), np.min(y_velocities), np.max(y_velocities)
    ]
    return np.array(features)

def extract_features(video_clips, threshold):
    return np.array([extract_features_from_median_series(track_median_points(clip, threshold)) for clip in video_clips])

def train():
    # Load your data
    fall_paths = glob.glob(os.path.join(falls_dir, "*.csv"))
    normal_paths = glob.glob(os.path.join(normal_dir, "*.csv"))
    fall_clips = [load_vid(path) for path in fall_paths]
    normal_clips = [load_vid(path) for path in normal_paths]
    video_clips = fall_clips + normal_clips
    labels = np.array([1]*len(fall_clips) + [0]*len(normal_clips))  # 1 for falls, 0 for normal

    # Extract features using the method described
    features = extract_features(video_clips, person_low)  # Using person_low as the threshold

    # Split the data into training and test sets with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, stratify=labels, random_state=42)

    # Calculate class weights
    class_weights = {0: 1., 1: len(normal_clips) / len(fall_clips)}

    # Train a Random Forest classifier with class weights
    model = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")

def classify_csv(csv_file_path, threshold=20):
    video = load_vid(csv_file_path)
    median_series = track_median_points(video, threshold)
    features = extract_features_from_median_series(median_series)
    features = features.reshape(1, -1)  # Reshape for a single sample prediction
    prediction = model.predict(features)
    return prediction

def test():
    test_files = glob.glob(os.path.join(test_dir, "*.csv"))
    results = {}
    for file_path in test_files:
        start_time = time.time()  # Start timing
        prediction = classify_csv(file_path, person_low)
        end_time = time.time()  # End timing
        inference_time = end_time - start_time  # Calculate inference time
        results[file_path] = ("Fall Detected" if prediction[0] == 1 else "Normal Activity", inference_time)
    return results


# Load your data
fall_paths = glob.glob(os.path.join(falls_dir, "*.csv"))
normal_paths = glob.glob(os.path.join(normal_dir, "*.csv"))
fall_clips = [load_vid(path) for path in fall_paths]
normal_clips = [load_vid(path) for path in normal_paths]
video_clips = fall_clips + normal_clips
labels = np.array([1]*len(fall_clips) + [0]*len(normal_clips))  # 1 for falls, 0 for normal

# Extract features using the method described
features = extract_features(video_clips, person_low)  # Using person_low as the threshold

# Split the data into training and test sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, stratify=labels, random_state=42)

# Calculate class weights
class_weights = {0: 1., 1: len(normal_clips) / len(fall_clips)}

# Train a Random Forest classifier with class weights
model = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")


# Save the model to disk
filename = 'finalized_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)


results = test()
for file, (result, time_taken) in results.items():
    print(f"{file}: {result} - Inference Time: {time_taken:.4f} seconds")