import cv2
import numpy as np
import requests
import base64
import time
import argparse
import sys
import signal
import io
from google.cloud import vision
from google.cloud.vision_v1 import types
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Configuration
GOOGLE_VISION_API_KEY = "AIzaSyBVt-BxOnkeWMaCBEJGglttwmpxuqV52rk"  # Replace with your actual Google Cloud Vision API key
VEHICLE_THRESHOLD = 15
DEFAULT_WAIT_TIME = 30
REDUCED_WAIT_TIME = 20
GOOGLE_VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

# Check if API key is still the placeholder
if GOOGLE_VISION_API_KEY == "your-google-vision-api-key":
    print("Error: Please replace 'your-google-vision-api-key' with your actual Google Cloud Vision API key.")
    sys.exit(1)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Vehicle Counting System")
parser.add_argument("--video_source", type=str, default="0", 
                    help="Video source: device index (e.g., 0) or stream URL")
parser.add_argument("--headless", action="store_true", help="Run without GUI display")
parser.add_argument("--use-local", action="store_true", help="Use local YOLO model instead of Google Vision API")
args = parser.parse_args()

# Initialize video capture with retry
def init_video_capture(source, retries=3, delay=5):
    for attempt in range(retries):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap
        print(f"Attempt {attempt + 1}/{retries}: Could not open video source {source}")
        time.sleep(delay)
    print(f"Error: Failed to open video source {source} after {retries} attempts")
    sys.exit(1)

# Determine video source
video_source = args.video_source
if video_source.isdigit():
    video_source = int(video_source)
cap = init_video_capture(video_source)

# Encode image to base64
def encode_image(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

# Test Google Vision API key
def test_api_key():
    headers = {"Content-Type": "application/json"}
    payload = {
        "requests": [
            {
                "image": {"content": encode_image(np.zeros((100, 100, 3), dtype=np.uint8))},
                "features": [{"type": "OBJECT_LOCALIZATION", "maxResults": 10}]
            }
        ]
    }
    try:
        print("Verifying Google Vision API key...")
        response = requests.post(f"{GOOGLE_VISION_ENDPOINT}?key={GOOGLE_VISION_API_KEY}", 
                                headers=headers, json=payload)
        response.raise_for_status()
        print("Google Vision API key verified.")
        return True
    except requests.exceptions.HTTPError as e:
        print(f"Error in Google Vision API call: {e}")
        if e.response.status_code == 401:
            print("Error 401: Unauthorized. Please check your API key.")
            print("Possible causes:")
            print("- Invalid or revoked API key. Generate a new key at https://console.cloud.google.com/apis/credentials")
            print("- Cloud Vision API not enabled. Enable it at https://console.cloud.google.com/apis/library/vision.googleapis.com")
        if e.response.content:
            print("Response details:", e.response.content.decode())
        print("Payload sent:", payload)
        return False

# Local YOLO-based vehicle counting
def count_vehicles_local(image):
    if not YOLO_AVAILABLE:
        print("Error: YOLO not available. Install ultralytics: pip install ultralytics")
        return 0
    try:
        model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 nano model
        results = model(image)
        vehicle_count = sum(1 for pred in results[0].boxes if pred.cls in [2, 7])  # Car, truck classes
        return vehicle_count
    except Exception as e:
        print(f"Error in local YOLO processing: {e}")
        return 0

# Count vehicles using Google Cloud Vision API
def count_vehicles_google_vision(image):
    headers = {"Content-Type": "application/json"}
    payload = {
        "requests": [
            {
                "image": {"content": encode_image(image)},
                "features": [{"type": "OBJECT_LOCALIZATION", "maxResults": 50}]
            }
        ]
    }
    try:
        response = requests.post(f"{GOOGLE_VISION_ENDPOINT}?key={GOOGLE_VISION_API_KEY}", 
                                headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        objects = result.get("responses", [{}])[0].get("localizedObjectAnnotations", [])
        # Count objects labeled as Vehicle, Car, or Truck
        vehicle_count = sum(1 for obj in objects if obj["name"].lower() in ["vehicle", "car", "truck"])
        return vehicle_count
    except requests.exceptions.HTTPError as e:
        print(f"Error in Google Vision API call: {e}")
        if e.response.content:
            print("Response details:", e.response.content.decode())
        print("Payload sent:", payload)
        return 0
    except (KeyError, ValueError) as e:
        print(f"Error parsing API response: {e}")
        return 0

# Wrapper for vehicle counting (API or local)
def count_vehicles(image):
    if args.use_local or not test_api_key():
        print("Using local YOLO model for vehicle counting.")
        return count_vehicles_local(image)
    return count_vehicles_google_vision(image)

# Adjust traffic light wait time
def adjust_traffic_light(vehicle_count):
    if vehicle_count > VEHICLE_THRESHOLD:
        print(f"Vehicle count ({vehicle_count}) exceeds threshold. Wait time: {REDUCED_WAIT_TIME}s")
        return REDUCED_WAIT_TIME
    else:
        print(f"Vehicle count ({vehicle_count}) within threshold. Wait time: {DEFAULT_WAIT_TIME}s")
        return DEFAULT_WAIT_TIME

# Signal handler for graceful exit
def signal_handler(sig, frame):
    print("\nExiting gracefully...")
    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Main function
def main():
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            vehicle_count = count_vehicles(frame)
            print(f"Detected {vehicle_count} vehicles.")

            wait_time = adjust_traffic_light(vehicle_count)

            if not args.headless:
                cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Wait Time: {wait_time}s", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Vehicle Counting", frame)

            time.sleep(1)  # Process every second

            if not args.headless and cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()