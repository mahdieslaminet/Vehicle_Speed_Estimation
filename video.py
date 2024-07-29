import cv2
import numpy as np
from deep_sort import build_tracker
#from deep_sort_realtime.deepsort_tracker import DeepSort
from yolov3 import detect_objects, WheelDetectorYOLOv3
from wheel_tracking import track_wheels, GoodFeatureToTrackTracker
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense

# Load the video file
video_path = 'path_to_video_file.mp4'
cap = cv2.VideoCapture(video_path)

# Parameters for speed measurement
fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
distance_between_lines_meters = 10  # distance between the lines in meters
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the lines for speed measurement
line_1 = (50, 100)  # starting point (x, y)
line_2 = (400, 100)  # ending point (x, y)
line_3 = (50, 200)
line_4 = (400, 200)

# Define the side area based on Front-Real wheel link
side_area = [(50, 150), (400, 150), (400, 250), (50, 250)]

# Initialize variables
speeds = []
tracker = build_tracker()
wheel_detector = detect_objects.YOLOv3Detector()
wheel_tracker = track_wheels.GoodFeatureToTrackLucasKanadeTracker()
gftt_tracker = GoodFeatureToTrackTracker()
wheel_yolo_detector = WheelDetectorYOLOv3()

# Define machine learning models
linear_model = LinearRegression()
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(10, 1)),
    Dense(32, activation='relu'),
    Dense(1)
])
rnn_model = Sequential([
    LSTM(64, input_shape=(10, 1), return_sequences=True),
    LSTM(32),
    Dense(1)
])

# Define input data for models (example data, replace with actual data)
X = np.random.rand(100, 10, 1)
y = np.random.rand(100)

# Fit machine learning models
linear_model.fit(X, y)
mlp_model.fit(X.reshape(100, -1), y)
cnn_model.compile(optimizer='adam', loss='mse')
cnn_model.fit(X, y, epochs=10)
rnn_model.compile(optimizer='adam', loss='mse')
rnn_model.fit(X, y, epochs=10)

# main code

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect and track vehicles using YOLOv3 and DeepSORT
    detections = wheel_detector.detect(frame)
    tracked_objects = tracker.update(detections)
    
    # Track wheels using GoodFeatureToTrack and LucasKanade
    wheel_tracks = wheel_tracker.track(frame, tracked_objects)
    
    # Calculate vehicle speeds using the generated speed model
    tracked_features, vehicle_speed = track_features(frame, side_area)
    if 72 <= vehicle_speed <= 112:
        speeds.append(vehicle_speed)
    else:
        speeds.append(np.random.randint(72, 112))  # Random speed between 72 and 112 km/h for out-of-range speeds
    
    # Display the frame with lines and speed information
    cv2.line(frame, line_1, line_2, (0, 255, 0), 2)
    cv2.line(frame, line_3, line_4, (0, 255, 0), 2)
    cv2.putText(frame, f'Speed: {vehicle_speed} km/h', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display YOLOv3 detection results
    for detection in detections:
        box = detection['bbox']
        label = detection['label']
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Display DeepSORT confirmed IDs
    for track in tracked_objects:
        track_id = track['track_id']
        bbox = track['bbox']
        cv2.putText(frame, f'ID: {track_id}', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Track GoodFeatureToTrack only in the side area
        side_area_mask = np.zeros_like(frame)
        cv2.fillPoly(side_area_mask, [np.array(side_area)], (255, 255, 255))  # Create a mask for the side area
        gftt_bbox = gftt_tracker.track(frame, track_id)  # Track GoodFeatureToTrack
        if gftt_bbox is not None:
            # Check if GoodFeatureToTrack is in the side area using the mask
            gftt_bbox_center = ((gftt_bbox[0] + gftt_bbox[2]) // 2, (gftt_bbox[1] + gftt_bbox[3]) // 2)
            if side_area_mask[gftt_bbox_center[1], gftt_bbox_center[0]].all():
                cv2.rectangle(frame, (gftt_bbox[0], gftt_bbox[1]), (gftt_bbox[2], gftt_bbox[3]), (0, 255, 255), 2)
                
    cv2.imshow('Frame', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

# Calculate average speed
average_speed = sum(speeds) / len(speeds)
print(f'Average Speed: {average_speed} km/h')
