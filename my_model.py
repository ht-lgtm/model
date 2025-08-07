import cv2
from collections import defaultdict
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
VIDEO_PATH = "C:/Users/SBA/Downloads/cut_cctv1.mp4"
MODEL_PATH = "C:/Users/SBA/github/model/best_300.pt"
STATIONARY_SECONDS_THRESHOLD = 30  # Seconds to wait before marking as illegal
POSITION_DEVIATION_THRESHOLD = 15  # Max pixels a car can move to be considered stationary

# --- Initialization ---
# Load the YOLOv11 model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure the model file exists at: {MODEL_PATH}")
    exit()


# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file at {VIDEO_PATH}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
# If FPS is 0, use a default value
if fps == 0:
    print("Warning: Video FPS is 0. Using a default of 30 FPS.")
    fps = 30

# Define the codec and create VideoWriter object
OUTPUT_PATH = "C:/Users/SBA/github/illegal_parking_output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

STATIONARY_FRAME_THRESHOLD = int(STATIONARY_SECONDS_THRESHOLD * fps)
print(f"Video FPS: {fps}")
print(f"Stationary frame threshold: {STATIONARY_FRAME_THRESHOLD} frames ({STATIONARY_SECONDS_THRESHOLD} seconds)")
print(f"Output video will be saved to: {OUTPUT_PATH}")


# Data storage
# Stores the history of center points for each track_id
track_history = defaultdict(list)
# Stores info about stationary state for each track_id
stationary_info = defaultdict(dict)
# Stores track_ids that are confirmed illegal parkers
illegal_parking_ids = set()

frame_number = 0

# --- Main Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video or failed to read frame.")
        break

    frame_number += 1
    
    # A simple progress indicator
    if frame_number % int(fps * 5) == 0: # Print progress every 5 seconds
        print(f"Processing frame {frame_number}...")


    # Run YOLO tracking on the frame
    # We are interested in 'car' (class 2), 'truck' (class 7), 'bus' (class 5) from COCO dataset
    results = model.track(frame, persist=True, classes=[0, 1], verbose=False, tracker="bytetrack.yaml")

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        confidences = results[0].boxes.conf.cpu().numpy()

        for box, track_id, class_id, confidence in zip(boxes, track_ids, class_ids, confidences):
            x1, y1, x2, y2 = box
            class_name = model.names[class_id]
            
            # Draw box and label
            if track_id in illegal_parking_ids:
                # This vehicle is already marked as illegally parked
                color = (0, 0, 255)  # Red
                label = f"{class_name} {track_id} - Illegally Parked (Conf: {confidence:.2f})"
            else:
                # This is a normally tracked vehicle
                color = (0, 255, 0)  # Green
                label = f"{class_name} ID: {track_id} (Conf: {confidence:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- Stationary Logic ---
            if track_id not in illegal_parking_ids:
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                current_pos = np.array([center_x, center_y])
                
                # Get the position history for this track
                history = track_history[track_id]
                history.append(current_pos)

                # Check if the object has moved significantly
                is_stationary = False
                if len(history) > 1:
                    # Compare current position to the first recorded position in this stationary window
                    start_pos = stationary_info[track_id].get('start_pos', history[0])
                    distance = np.linalg.norm(current_pos - start_pos)
                    
                    if distance < POSITION_DEVIATION_THRESHOLD:
                        is_stationary = True
                        # If it just became stationary, record the start frame and position
                        if 'start_frame' not in stationary_info[track_id]:
                            stationary_info[track_id]['start_frame'] = frame_number
                            stationary_info[track_id]['start_pos'] = current_pos
                    
                if is_stationary:
                    # If it is stationary, check if it has been so for long enough
                    duration_frames = frame_number - stationary_info[track_id]['start_frame']
                    if duration_frames >= STATIONARY_FRAME_THRESHOLD:
                        illegal_parking_ids.add(track_id)
                        print(f"Vehicle with ID {track_id} marked as illegally parked.")
                        # Clean up data for this ID as it's now permanently marked
                        del track_history[track_id]
                        del stationary_info[track_id]
                else:
                    # If it moved, reset its history and stationary info
                    track_history[track_id] = [current_pos] # Keep only the latest position
                    if track_id in stationary_info:
                        del stationary_info[track_id]

    # Write the frame to the output video file
    out.write(frame)


# --- Cleanup ---
cap.release()
out.release()
print(f"Processing finished. The output video is saved at: {OUTPUT_PATH}")
