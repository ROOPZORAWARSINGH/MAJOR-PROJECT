import cv2
import mediapipe as mp
import numpy as np

from function import *
from keras.models import model_from_json

# Load the model architecture from JSON file
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()

# Create the model from the loaded architecture
model = model_from_json(model_json)

# Load the model weights
model.load_weights("model.h5")

# Define colors for visualization
colors = [(245, 117, 16) for _ in range(20)]

# Define a threshold for prediction
threshold = 0.8

# Initialize variables
sequence = []
sentence = []
accuracy = []
predictions = []

# Define output list
output_list = []

# Start video capture
cap = cv2.VideoCapture(0)

# Set mediapipe model parameters
with mp.solutions.hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Crop frame to desired size and draw a rectangle
        cropframe = frame[40:400, 0:400]
        frame = cv2.rectangle(frame, (0, 40), (400, 400), (255, 255, 255), 2)

        # Perform hand detection
        image, results = mediapipe_detection(cropframe, hands)

        # Extract keypoints and update sequence
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-15:]

        try:
            if len(sequence) == 15:
                # Perform prediction
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_action = actions[np.argmax(res)]
                print(predicted_action)
                predictions.append(predicted_action)

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if predicted_action != sentence[-1]:
                                sentence.append(predicted_action)
                                accuracy.append(str(res[np.argmax(res)] * 100))
                        else:
                            sentence.append(predicted_action)
                            accuracy.append(str(res[np.argmax(res)] * 100))

                if len(sentence) > 1:
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]

                # Store output one by one
                output_text = ' '.join(sentence) + ' '.join(accuracy)
                output_list.append(output_text)

        except Exception as e:
            pass

        # Draw output text
        cv2.rectangle(frame, (0, 0), (400, 40), (245, 117, 16), -1)
        cv2.putText(frame, "Output: -" + output_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('OpenCV Feed', frame)

        # Break loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    print(output_list)