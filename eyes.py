import cv2
import numpy as np
import os
import logging

# Configure logging for better error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for cascade file paths - use OpenCV's built-in data directory
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"
DOWNSCALE_FACTOR = 0.5
EYE_RADIUS = 10  # Define the radius for eye circles

def load_cascade_classifier(path, name):
    """
    Load a Haar cascade classifier with error handling.

    Args:
        path (str): Path to the cascade XML file.
        name (str): Name of the classifier for error messages.

    Returns:
        cv2.CascadeClassifier: Loaded classifier.

    Raises:
        FileNotFoundError: If the cascade file does not exist.
        ValueError: If the cascade file cannot be loaded.
    """
    if not os.path.exists(path):
        logging.error(f"Cascade file not found: {path}")
        raise FileNotFoundError(f"Unable to find {name} cascade classifier XML file at {path}")

    classifier = cv2.CascadeClassifier(path)
    if classifier.empty():
        logging.error(f"Failed to load {name} cascade classifier from {path}")
        raise ValueError(f"Unable to load {name} cascade classifier XML file from {path}")

    logging.info(f"Successfully loaded {name} cascade classifier")
    return classifier

def detect_eyes_in_frame(frame, face_cascade, eye_cascade, ds_factor):
    """
    Detect faces and eyes in a frame.

    Args:
        frame: Input frame from camera.
        face_cascade: Face cascade classifier.
        eye_cascade: Eye cascade classifier.
        ds_factor: Downscale factor for performance.

    Returns:
        Processed frame with eye detections marked.
    """
    # Downscale for performance
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))

        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            center = (int(x_eye + 0.5 * w_eye), int(y_eye + 0.5 * h_eye))
            cv2.circle(roi_color, center, EYE_RADIUS, (0, 255, 0), 3)

    return frame

def main():
    """Main function to run the eye detection application."""
    try:
        # Load cascade classifiers with error handling
        face_cascade = load_cascade_classifier(FACE_CASCADE_PATH, "face")
        eye_cascade = load_cascade_classifier(EYE_CASCADE_PATH, "eye")

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Unable to access the camera")
            raise RuntimeError("Unable to access the camera. Please check camera connection.")

        logging.info("Camera initialized successfully")

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame from camera")
                break

            # Process frame for eye detection
            processed_frame = detect_eyes_in_frame(frame, face_cascade, eye_cascade, DOWNSCALE_FACTOR)

            # Display the result
            cv2.imshow('Eye Detector', processed_frame)

            # Exit on ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    finally:
        # Clean up resources
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
            