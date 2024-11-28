import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Access the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot Open Camera")
    exit()

# Setup hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load model
model = load_model("digit_recognizer.h5")

# ROI coordinates (adjust as needed)
roi_x_start, roi_y_start = 1000, 300  # Top-left corner of the ROI
roi_width, roi_height = 400, 400     # Width and height of the ROI

# Function to preprocess the ROI for prediction
def preprocess_roi(frame):
    roi = frame[roi_y_start:roi_y_start + roi_height, roi_x_start:roi_x_start + roi_width]

    # Resize to 28x28
    roi_resized = cv2.resize(roi, (28, 28))

    # Convert to grayscale
    if len(roi_resized.shape) == 3:  # Only convert if it's in BGR
        roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi_resized

    # Normalize to [0, 1]
    roi_normalized = roi_gray.astype('float32') / 255.0
    
    # Reshape for the model input
    roi_normalized = np.expand_dims(roi_normalized, axis=[0, -1])  # (1, 28, 28, 1)

    return roi_normalized

# Function to check if lines should be drawn, point with pointer finger and pinch thumb and middle finger together
def write(hand_landmarks):
    # Define fingertip and base indices for fingers
    fingertip_indices = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    base_indices = [2, 6, 10, 14, 18]


    middle_tip = hand_landmarks.landmark[fingertip_indices[2]]
    thumb_tip = hand_landmarks.landmark[fingertip_indices[0]]

    # Check if the middle and thumb touching
    distance = np.sqrt((thumb_tip.x - middle_tip.x)**2 + (thumb_tip.y - middle_tip.y)**2)
    if distance > 0.05: 
        return False
    

    index_tip = hand_landmarks.landmark[fingertip_indices[1]]

    # Check if the index finger is extended
    index_distance = np.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
    if index_distance < 0.1:  # Index finger not extended
        return False

    return True

# Function to clear the drawings by pinch all 5 fingers together
def clear(hand_landmarks):
    """
    Checks if all fingertips (index, middle, ring, pinky) are close to the thumb tip.
    """
    # Fingertip indices
    thumb_tip = 4
    other_fingertips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky

    # Get thumb tip position
    thumb = hand_landmarks.landmark[thumb_tip]

    # Check distances from each fingertip to the thumb tip
    for idx in other_fingertips:
        fingertip = hand_landmarks.landmark[idx]
        distance = np.sqrt(
            (thumb.x - fingertip.x) ** 2 +
            (thumb.y - fingertip.y) ** 2 +
            (thumb.z - fingertip.z) ** 2
        )
    
        if distance > 0.04:  # Threshold (adjust as needed)
            return False
    return True


persistent_points = []
shapes = [[]]
start_new_shape = True



while True:
    # Step 2: Capture frame-by-frame
    succesfulCapture, frame = cap.read()
    if not succesfulCapture:
        print("Error: Failed to grab a frame.")
        break

    frame = cv2.flip(frame, 1)

    # Introduce frame to draw within for model prediction
    cv2.rectangle(frame, 
                  (roi_x_start, roi_y_start), 
                  (roi_x_start + roi_width, roi_y_start + roi_height), 
                  (255, 0, 0), 2)  # Blue rectangle with thickness 2
    
    # Convert the frame to RGB (MediaPipe works with RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            savedx = -1
            savedy = -1
            for i, landmark in enumerate(hand_landmarks.landmark):
                # Convert normalized coordinates to pixel values
                height, width, _ = frame.shape
                x, y = int(landmark.x * width), int(landmark.y * height)

                # Overlay points on key landmarks (e.g., fingertips)
                if i in [4, 8, 12, 16, 20]:  # Fingertips indices
                    cv2.circle(frame, (x, y), radius=8, color=(0, 255, 0), thickness=-1)  # Green circle

                if i == 8:
                    #persistent_points.append((x,y))
                    savedx = x
                    savedy = y

            if write(hand_landmarks):
                if start_new_shape:
                    shapes.append([])  # Add a new empty shape
                    start_new_shape = False
                shapes[-1].append((savedx, savedy))
            else:
                start_new_shape = True
            if clear(hand_landmarks):
                shapes.clear()

    for shape in shapes:
        for i in range(1, len(shape)):
            cv2.line(frame, shape[i - 1], shape[i], color=(0, 0, 0), thickness=20)


    # Preprocess ROI and predict
    processed_roi = preprocess_roi(frame)
    # cv2.imshow('Processed frame', processed_roi.reshape(28, 28) * 255)
    prediction = model.predict(processed_roi)
    predicted_digit = np.argmax(prediction)

    # Display the prediction
    cv2.putText(frame, f"Predicted: {predicted_digit}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)


    # Step 3: Display the current frame
    cv2.imshow('Camera Feed', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Press 'c' to clear points
        shapes.clear()
    elif key == ord('q'):  # Quit program
        break

# Step 5: Release the camera and close windows
cap.release()
cv2.destroyAllWindows()