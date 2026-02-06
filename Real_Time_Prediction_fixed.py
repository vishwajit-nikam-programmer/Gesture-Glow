import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import logging
import os
import collections
from collections import deque

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()
    ]
)

class GesturePredictor:
    def __init__(self, confidence_threshold=0.8, smoothing_window=7):
        """
        Initialize gesture predictor with smoothing and confidence threshold
        Args:
            confidence_threshold: Minimum confidence required for prediction
            smoothing_window: Number of frames to consider for smoothing
        """
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.gesture_buffer = deque(maxlen=smoothing_window)
        self.current_gesture = None
        self.stable_count = 0
        self.stable_threshold = 3  # Number of stable predictions needed
        
    def get_prediction(self, prediction, confidence):
        """
        Get smoothed gesture prediction
        Args:
            prediction: Raw prediction probabilities
            confidence: Prediction confidence
        Returns:
            Tuple of (gesture, confidence)
        """
        predicted_class = np.argmax(prediction)
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            return "Unknown", 0.0
        
        # Add to buffer and apply smoothing
        self.gesture_buffer.append(predicted_class)
        
        # Check for stable prediction
        if len(self.gesture_buffer) == self.smoothing_window:
            if len(set(self.gesture_buffer)) == 1:
                self.stable_count += 1
                if self.stable_count >= self.stable_threshold:
                    self.current_gesture = predicted_class
                    self.stable_count = 0
            else:
                self.stable_count = 0
        
        # Return current stable gesture or "Unstable" if not stable
        if self.current_gesture is not None:
            return self.current_gesture, confidence
        return "Unstable", confidence

def preprocess_image(img, target_size=224, augment=False):
    """
    Preprocess image for model input with optional augmentation
    Args:
        img: Input image (can be any size)
        target_size: Target size for model input
        augment: Whether to apply data augmentation
    Returns:
        Preprocessed image
    """
    if img is None:
        return None
    
    # Convert to RGB if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE for better contrast
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Maintain aspect ratio and pad
    h, w = img.shape[:2]
    aspect_ratio = w / h
    
    if aspect_ratio > 1:
        new_w = target_size
        new_h = int(target_size / aspect_ratio)
    else:
        new_h = target_size
        new_w = int(target_size * aspect_ratio)
    
    img = cv2.resize(img, (new_w, new_h))
    
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # Optional augmentation
    if augment:
        # Random brightness adjustment
        alpha = 1.0 + np.random.uniform(-0.1, 0.1)
        beta = np.random.uniform(-20, 20)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Random rotation
        angle = np.random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((target_size//2, target_size//2), angle, 1)
        img = cv2.warpAffine(img, M, (target_size, target_size))
    
    return img

def draw_prediction(frame, prediction, confidence, fps):
    """
    Draw prediction results on frame
    Args:
        frame: Input frame
        prediction: Gesture prediction
        confidence: Prediction confidence
        fps: Current FPS
    """
    # Create a semi-transparent overlay
    overlay = frame.copy()
    
    # Draw prediction box with different colors based on prediction
    colors = {
        'lights_on': (0, 255, 0),    # Green
        'lights_off': (0, 0, 255),   # Red
        'fan_on': (255, 165, 0),     # Orange
        'fan_off': (255, 0, 255),    # Purple
        'Unknown': (0, 0, 0),         # Black
        'Unstable': (255, 255, 0)    # Yellow
    }
    
    color = colors.get(prediction, (0, 0, 0))
    
    # Draw prediction box
    cv2.rectangle(overlay, (10, 10), (300, 100), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw prediction text
    cv2.putText(frame, f"Gesture: {prediction}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}%", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def main():
    try:
        # Define the gesture model directory
        model_dir = r"C:\Users\nikam\Music\Final Code\Gesture Models"

        # Find all model files with .keras or .h5 extension
        model_files = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5'))]
        if not model_files:
            raise FileNotFoundError("No model file found in the specified directory")

        # Select the most recently modified model file
        model_files_paths = [os.path.join(model_dir, f) for f in model_files]
        latest_model_path = max(model_files_paths, key=os.path.getmtime)

        logging.info(f"Loading model from: {latest_model_path}")
        model = load_model(latest_model_path)

        # Initialize gesture predictor
        gesture_predictor = GesturePredictor(confidence_threshold=0.8, smoothing_window=7)

        # Load preprocessing parameters
        preprocessing_params = {
            'target_size': 224,
            'classes': ['fan_off', 'fan_on', 'lights_off', 'lights_on'],
            'confidence_threshold': 0.8
        }

        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam")

        # Camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set desired FPS

        frame_count, start_time, fps = 0, time.time(), 0
        logging.info("Starting real-time gesture prediction. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to capture frame")
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            # Update FPS every 30 frames
            if frame_count >= 30:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()

            # Preprocess frame
            processed_frame = preprocess_image(frame, preprocessing_params['target_size'])
            if processed_frame is None:
                continue

            input_frame = np.expand_dims(processed_frame / 255.0, axis=0)
            prediction = model.predict(input_frame, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]
            
            # Get smoothed prediction
            gesture, confidence = gesture_predictor.get_prediction(prediction, confidence)
            
            # Convert gesture index to class name
            if gesture != "Unknown" and gesture != "Unstable":
                class_name = preprocessing_params['classes'][gesture]
            else:
                class_name = gesture

            # Draw results
            draw_prediction(frame, class_name, confidence * 100, fps)

            # Show small processed frame
            small_processed = cv2.resize(processed_frame, (160, 160))
            frame[10:170, frame.shape[1] - 170:frame.shape[1] - 10] = cv2.cvtColor(small_processed, cv2.COLOR_RGB2BGR)

            cv2.imshow("Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        logging.info("Prediction stopped")

if __name__ == "__main__":
    main()
