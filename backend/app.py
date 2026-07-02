from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import time
import threading
from ultralytics import YOLO
import easyocr
import logging
import re

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

VIDEO_PATH = "1.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    logger.error(f"Cannot open video file: {VIDEO_PATH}")
    VIDEO_PATH = 0
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error("Cannot open webcam either")
else:
    logger.info(f"Successfully opened video: {VIDEO_PATH}")

try:
    model = YOLO("yolov8n.pt")
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    model = None

try:
    plate_model = YOLO("license_plate_detector.pt")
    logger.info("License plate detection model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load license plate detection model: {e}")
    logger.info("Using EasyOCR for text detection instead")
    plate_model = None

try:
    reader = easyocr.Reader(["en"])
    logger.info("EasyOCR model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load EasyOCR model: {e}")
    reader = None

vehicle_classes = {
    2: "car",
    5: "bus",
    7: "truck",
    3: "motorcycle",
    1: "bicycle"
}

vehicle_count = 0
wait_time = 25

vehicle_type_counts = {
    "car": 0,
    "bus": 0,
    "truck": 0,
    "motorcycle": 0,
    "bicycle": 0
}

congestion_score = 0
congestion_level = "Low"
ai_recommendation = "Traffic flow normal. Maintain current signal cycle."

predicted_vehicle_count = 0
traffic_trend = "Stable"
traffic_history = []
last_history_update = 0

emergency_mode = False
emergency_message = "No emergency vehicle detected."

accident_mode = False
accident_message = "No accident or road blockage detected."

signal_mode = "Normal"
safety_status = "Safe"
system_mode = "AI Monitoring Active"
ai_confidence = 92

detected_plates = []
detection_data = {"vehicles": [], "plates": []}

last_plate_detection_time = 0
plate_detection_interval = 2


def clean_plate_text(text):
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", text.upper())
    if 5 <= len(cleaned) <= 12:
        return cleaned
    return None


def detect_license_plates_yolo(frame):
    plates = []
    plate_boxes = []

    try:
        if plate_model:
            results = plate_model(frame)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])

                    if confidence > 0.5:
                        plate_region = frame[y1:y2, x1:x2]

                        if reader and plate_region.size > 0:
                            rgb_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB)
                            ocr_results = reader.readtext(rgb_plate)

                            for bbox, text, conf in ocr_results:
                                if conf > 0.6:
                                    cleaned_text = clean_plate_text(text)

                                    if cleaned_text:
                                        plates.append(cleaned_text)
                                        plate_boxes.append({
                                            "text": cleaned_text,
                                            "bbox": [x1, y1, x2, y2],
                                            "confidence": float(conf)
                                        })

    except Exception as e:
        logger.error(f"Error in YOLO license plate detection: {e}")

    return frame, plates, plate_boxes


def detect_license_plates_easyocr(frame):
    plates = []
    plate_boxes = []

    try:
        if reader:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = reader.readtext(rgb_frame)

            for bbox, text, confidence in results:
                if confidence > 0.6:
                    cleaned_text = clean_plate_text(text)

                    if cleaned_text:
                        plates.append(cleaned_text)
                        top_left = tuple(map(int, bbox[0]))
                        bottom_right = tuple(map(int, bbox[2]))

                        plate_boxes.append({
                            "text": cleaned_text,
                            "bbox": [top_left[0], top_left[1], bottom_right[0], bottom_right[1]],
                            "confidence": float(confidence)
                        })

    except Exception as e:
        logger.error(f"Error in EasyOCR license plate detection: {e}")

    return frame, plates, plate_boxes


def update_traffic_prediction():
    global predicted_vehicle_count, traffic_trend, traffic_history, last_history_update

    now = time.time()

    if now - last_history_update < 2:
        return

    last_history_update = now

    traffic_history.append({
        "time": time.strftime("%H:%M:%S"),
        "count": vehicle_count,
        "congestion": congestion_score
    })

    traffic_history = traffic_history[-20:]

    if len(traffic_history) < 4:
        predicted_vehicle_count = vehicle_count
        traffic_trend = "Stable"
        return

    recent = [item["count"] for item in traffic_history[-5:]]
    previous = [item["count"] for item in traffic_history[-10:-5]] or recent

    recent_avg = sum(recent) / len(recent)
    previous_avg = sum(previous) / len(previous)

    trend_delta = recent_avg - previous_avg

    if trend_delta > 1:
        traffic_trend = "Increasing"
    elif trend_delta < -1:
        traffic_trend = "Decreasing"
    else:
        traffic_trend = "Stable"

    predicted_vehicle_count = max(0, int(recent_avg + trend_delta))


def calculate_traffic_analytics():
    global congestion_score, congestion_level, wait_time, ai_recommendation
    global signal_mode, safety_status, system_mode, ai_confidence

    weighted_density = (
        vehicle_type_counts["car"] * 1 +
        vehicle_type_counts["motorcycle"] * 0.5 +
        vehicle_type_counts["bicycle"] * 0.3 +
        vehicle_type_counts["bus"] * 2.5 +
        vehicle_type_counts["truck"] * 2.2
    )

    congestion_score = min(100, int(weighted_density * 8))
    heavy_vehicle_count = vehicle_type_counts["bus"] + vehicle_type_counts["truck"]

    update_traffic_prediction()

    if accident_mode:
        congestion_level = "Incident Alert"
        signal_mode = "Incident Alert"
        safety_status = "Road Risk"
        system_mode = "Traffic Diversion Mode"
        ai_confidence = 96
        wait_time = 65
        ai_recommendation = (
            "Accident or road blockage alert activated. Signal timing adjusted to support traffic diversion. "
            "Recommended action: notify traffic control, reduce incoming flow, and prioritize clearing the blocked lane."
        )
    elif emergency_mode:
        congestion_level = "Emergency Priority"
        signal_mode = "Emergency Priority"
        safety_status = "Priority Vehicle"
        system_mode = "Green Corridor Mode"
        ai_confidence = 98
        wait_time = 75
        ai_recommendation = (
            "Emergency vehicle priority activated. Green corridor enabled for 75 seconds. "
            "Reason: ambulance/fire/police priority mode was triggered for safe and faster movement."
        )
    elif congestion_score >= 70:
        congestion_level = "High"
        signal_mode = "Congestion Control"
        safety_status = "Congested"
        system_mode = "Adaptive Signal Mode"
        ai_confidence = 94
        wait_time = 55
        ai_recommendation = (
            f"High congestion detected. Extend green signal to {wait_time} seconds. "
            f"Reason: {vehicle_count} vehicles detected with {heavy_vehicle_count} heavy vehicles. "
            f"Traffic trend is {traffic_trend.lower()} with predicted count {predicted_vehicle_count}."
        )
    elif congestion_score >= 35:
        congestion_level = "Medium"
        signal_mode = "Congestion Control"
        safety_status = "Moderate Flow"
        system_mode = "Adaptive Signal Mode"
        ai_confidence = 91
        wait_time = 40
        ai_recommendation = (
            f"Moderate traffic detected. Set green signal to {wait_time} seconds. "
            f"Traffic trend is {traffic_trend.lower()}."
        )
    else:
        congestion_level = "Low"
        signal_mode = "Normal"
        safety_status = "Safe"
        system_mode = "AI Monitoring Active"
        ai_confidence = 89
        wait_time = 25
        ai_recommendation = (
            f"Traffic is smooth. Maintain green signal at {wait_time} seconds. "
            f"Traffic trend is {traffic_trend.lower()}."
        )


def process_frames():
    global cap, vehicle_count, model, detected_plates, last_plate_detection_time, detection_data
    global vehicle_type_counts

    while True:
        if not cap.isOpened():
            cap = cv2.VideoCapture(VIDEO_PATH)
            if not cap.isOpened():
                logger.error("Failed to reopen video")
                time.sleep(2)
                continue

        ret, frame = cap.read()

        if not ret:
            logger.warning("End of video reached, restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        current_time = time.time()
        frame_height, frame_width = frame.shape[:2]
        new_plates = []
        vehicle_detections = []
        plate_detections = []

        if model is not None:
            try:
                results = model(frame)

                vehicle_count = 0
                vehicle_type_counts = {
                    "car": 0,
                    "bus": 0,
                    "truck": 0,
                    "motorcycle": 0,
                    "bicycle": 0
                }

                for pred in results[0].boxes:
                    class_id = int(pred.cls[0])

                    if class_id in vehicle_classes:
                        vehicle_count += 1
                        vehicle_type = vehicle_classes[class_id]
                        vehicle_type_counts[vehicle_type] += 1

                        x1, y1, x2, y2 = map(int, pred.xyxy[0])
                        confidence = float(pred.conf[0])

                        vehicle_detections.append({
                            "class": vehicle_type,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence
                        })

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{vehicle_type} {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                calculate_traffic_analytics()

                if current_time - last_plate_detection_time >= plate_detection_interval:
                    if plate_model:
                        processed_frame, plates, plate_boxes = detect_license_plates_yolo(frame.copy())
                    else:
                        processed_frame, plates, plate_boxes = detect_license_plates_easyocr(frame.copy())

                    new_plates = plates
                    plate_detections = plate_boxes
                    last_plate_detection_time = current_time

            except Exception as e:
                logger.error(f"Error in vehicle detection: {e}")

        detection_data = {
            "vehicles": vehicle_detections,
            "plates": plate_detections,
            "frame_size": {"width": frame_width, "height": frame_height}
        }

        for plate in new_plates:
            if plate not in detected_plates:
                detected_plates.append(plate)
                logger.info(f"Detected new license plate: {plate}")

        detected_plates = detected_plates[-20:]
        time.sleep(0.1)


threading.Thread(target=process_frames, daemon=True).start()


def generate_frames():
    global cap

    while True:
        if not cap.isOpened():
            cap = cv2.VideoCapture(VIDEO_PATH)
            if not cap.isOpened():
                logger.error("Cannot open video for streaming")
                time.sleep(1)
                continue

        success, frame = cap.read()

        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 360))

        cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Signal Time: {wait_time}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Congestion: {congestion_level} {congestion_score}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Predicted: {predicted_vehicle_count} | Trend: {traffic_trend}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Cars:{vehicle_type_counts['car']} Bus:{vehicle_type_counts['bus']} Truck:{vehicle_type_counts['truck']}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if accident_mode:
            cv2.putText(frame, "INCIDENT ALERT: ACCIDENT / ROAD BLOCKAGE", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif emergency_mode:
            cv2.putText(frame, "EMERGENCY PRIORITY: GREEN CORRIDOR ACTIVE", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if detected_plates:
            cv2.putText(frame, f"Latest: {detected_plates[-1]}", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/traffic-data")
def traffic_data():
    return jsonify({
        "vehicle_count": vehicle_count,
        "vehicle_types": vehicle_type_counts,
        "congestion_score": congestion_score,
        "congestion_level": congestion_level,
        "wait_time": wait_time,
        "ai_recommendation": ai_recommendation,
        "predicted_vehicle_count": predicted_vehicle_count,
        "traffic_trend": traffic_trend,
        "traffic_history": traffic_history,
        "emergency_mode": emergency_mode,
        "emergency_message": emergency_message,
        "accident_mode": accident_mode,
        "accident_message": accident_message,
        "signal_mode": signal_mode,
        "safety_status": safety_status,
        "system_mode": system_mode,
        "ai_confidence": ai_confidence,
        "plates": detected_plates,
        "detections": detection_data
    })


@app.route("/api/trigger-emergency", methods=["POST"])
def trigger_emergency():
    global emergency_mode, emergency_message
    emergency_mode = True
    emergency_message = "Emergency vehicle detected. Green corridor activated."
    calculate_traffic_analytics()
    return jsonify({
        "status": "success",
        "emergency_mode": emergency_mode,
        "message": emergency_message
    })


@app.route("/api/clear-emergency", methods=["POST"])
def clear_emergency():
    global emergency_mode, emergency_message
    emergency_mode = False
    emergency_message = "No emergency vehicle detected."
    calculate_traffic_analytics()
    return jsonify({
        "status": "success",
        "emergency_mode": emergency_mode,
        "message": emergency_message
    })


@app.route("/api/trigger-accident", methods=["POST"])
def trigger_accident():
    global accident_mode, accident_message
    accident_mode = True
    accident_message = "Accident or road blockage detected. Traffic diversion recommended."
    calculate_traffic_analytics()
    return jsonify({
        "status": "success",
        "accident_mode": accident_mode,
        "message": accident_message
    })


@app.route("/api/clear-accident", methods=["POST"])
def clear_accident():
    global accident_mode, accident_message
    accident_mode = False
    accident_message = "No accident or road blockage detected."
    calculate_traffic_analytics()
    return jsonify({
        "status": "success",
        "accident_mode": accident_mode,
        "message": accident_message
    })


@app.route("/api/clear-plates")
def clear_plates():
    global detected_plates
    detected_plates = []
    return jsonify({"message": "Plates cleared", "plates": detected_plates})


@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>VisionFlow AI | Mobility Operations Center</title>
        <style>
            :root {
                --bg: #05070d;
                --panel: rgba(12, 18, 32, 0.88);
                --panel-strong: rgba(15, 23, 42, 0.96);
                --panel-soft: rgba(30, 41, 59, 0.72);
                --line: rgba(148, 163, 184, 0.18);
                --line-strong: rgba(56, 189, 248, 0.32);
                --text: #e5edf7;
                --muted: #8fa3bd;
                --muted-2: #64748b;
                --cyan: #38bdf8;
                --cyan-soft: rgba(56, 189, 248, 0.14);
                --green: #22c55e;
                --green-soft: rgba(34, 197, 94, 0.14);
                --amber: #f59e0b;
                --amber-soft: rgba(245, 158, 11, 0.16);
                --red: #ef4444;
                --red-soft: rgba(239, 68, 68, 0.16);
                --purple: #a78bfa;
                --purple-soft: rgba(167, 139, 250, 0.14);
                --radius-lg: 22px;
                --radius-md: 16px;
                --radius-sm: 12px;
                --shadow: 0 22px 70px rgba(0, 0, 0, 0.45);
            }

            * { box-sizing: border-box; }

            html { scroll-behavior: smooth; }

            body {
                margin: 0;
                min-height: 100vh;
                color: var(--text);
                font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
                background:
                    radial-gradient(circle at 16% 8%, rgba(14, 165, 233, 0.23), transparent 24%),
                    radial-gradient(circle at 78% 4%, rgba(45, 212, 191, 0.12), transparent 26%),
                    radial-gradient(circle at 70% 80%, rgba(124, 58, 237, 0.13), transparent 28%),
                    linear-gradient(135deg, #030712 0%, #07111f 44%, #0f172a 100%);
                overflow-x: hidden;
            }

            body::before {
                content: "";
                position: fixed;
                inset: 0;
                pointer-events: none;
                background-image:
                    linear-gradient(rgba(148, 163, 184, 0.04) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(148, 163, 184, 0.04) 1px, transparent 1px);
                background-size: 42px 42px;
                mask-image: linear-gradient(to bottom, rgba(0,0,0,0.9), rgba(0,0,0,0.08));
            }

            .app-shell {
                min-height: 100vh;
                display: grid;
                grid-template-columns: 280px minmax(0, 1fr);
            }

            .sidebar {
                position: sticky;
                top: 0;
                height: 100vh;
                padding: 24px 18px;
                background: rgba(2, 6, 23, 0.78);
                border-right: 1px solid var(--line);
                backdrop-filter: blur(18px);
                z-index: 4;
            }

            .brand {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 12px 10px 22px;
                border-bottom: 1px solid var(--line);
                margin-bottom: 18px;
            }

            .brand-mark {
                width: 44px;
                height: 44px;
                border-radius: 14px;
                display: grid;
                place-items: center;
                background: linear-gradient(135deg, rgba(56, 189, 248, 0.9), rgba(34, 197, 94, 0.78));
                box-shadow: 0 0 34px rgba(56, 189, 248, 0.32);
                font-weight: 900;
                color: #03121d;
                letter-spacing: -0.08em;
            }

            .brand h1 {
                margin: 0;
                font-size: 20px;
                letter-spacing: -0.04em;
            }

            .brand p {
                margin: 3px 0 0;
                color: var(--muted);
                font-size: 12px;
                line-height: 1.35;
            }

            .nav-label {
                color: var(--muted-2);
                font-size: 11px;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                margin: 18px 12px 8px;
            }

            .nav-item {
                display: flex;
                align-items: center;
                gap: 11px;
                color: #cbd5e1;
                text-decoration: none;
                padding: 11px 12px;
                border-radius: 14px;
                margin-bottom: 6px;
                border: 1px solid transparent;
                transition: 0.18s ease;
            }

            .nav-item:hover, .nav-item.active {
                background: rgba(15, 23, 42, 0.86);
                border-color: var(--line);
                color: #f8fafc;
                transform: translateX(2px);
            }

            .nav-icon {
                width: 28px;
                height: 28px;
                display: grid;
                place-items: center;
                border-radius: 10px;
                background: rgba(56, 189, 248, 0.1);
                color: var(--cyan);
                font-size: 14px;
            }

            .sidebar-footer {
                position: absolute;
                left: 18px;
                right: 18px;
                bottom: 18px;
                padding: 14px;
                border-radius: 16px;
                background: linear-gradient(135deg, rgba(14, 165, 233, 0.14), rgba(34, 197, 94, 0.10));
                border: 1px solid var(--line-strong);
            }

            .sidebar-footer strong {
                display: block;
                font-size: 13px;
                margin-bottom: 5px;
            }

            .sidebar-footer span {
                display: block;
                color: var(--muted);
                font-size: 12px;
                line-height: 1.45;
            }

            .main {
                padding: 24px;
                position: relative;
                z-index: 1;
            }

            .topbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 18px;
                margin-bottom: 22px;
            }

            .hero h2 {
                margin: 0;
                font-size: clamp(27px, 4vw, 44px);
                letter-spacing: -0.055em;
                line-height: 1.04;
            }

            .hero p {
                margin: 9px 0 0;
                color: var(--muted);
                max-width: 920px;
                font-size: 14px;
                line-height: 1.6;
            }

            .ops-strip {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                align-items: center;
                justify-content: flex-end;
            }

            .pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                white-space: nowrap;
                border-radius: 999px;
                padding: 9px 12px;
                border: 1px solid var(--line);
                background: rgba(15, 23, 42, 0.72);
                color: #cbd5e1;
                font-size: 12px;
                font-weight: 700;
            }

            .pulse-dot {
                width: 9px;
                height: 9px;
                border-radius: 50%;
                background: var(--green);
                box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7);
                animation: pulse 1.8s infinite;
            }

            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.6); }
                70% { box-shadow: 0 0 0 11px rgba(34, 197, 94, 0); }
                100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
            }

            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 14px;
                margin-bottom: 18px;
            }

            .kpi-card, .panel, .mini-panel {
                background: var(--panel);
                border: 1px solid var(--line);
                box-shadow: var(--shadow);
                backdrop-filter: blur(18px);
            }

            .kpi-card {
                position: relative;
                overflow: hidden;
                border-radius: var(--radius-md);
                padding: 18px;
                min-height: 138px;
            }

            .kpi-card::after {
                content: "";
                position: absolute;
                inset: auto -24px -40px auto;
                width: 110px;
                height: 110px;
                background: radial-gradient(circle, rgba(56, 189, 248, 0.16), transparent 68%);
            }

            .kpi-head {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 18px;
            }

            .kpi-label {
                color: var(--muted);
                font-size: 12px;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .kpi-icon {
                width: 34px;
                height: 34px;
                display: grid;
                place-items: center;
                border-radius: 12px;
                background: var(--cyan-soft);
                color: var(--cyan);
            }

            .kpi-value {
                font-size: 34px;
                line-height: 1;
                font-weight: 900;
                letter-spacing: -0.06em;
            }

            .kpi-sub {
                margin-top: 9px;
                color: var(--muted);
                font-size: 12px;
                line-height: 1.45;
            }

            .layout-grid {
                display: grid;
                grid-template-columns: minmax(0, 1.62fr) minmax(360px, 0.95fr);
                gap: 18px;
                align-items: start;
            }

            .left-stack, .right-stack {
                display: grid;
                gap: 18px;
            }

            .panel {
                border-radius: var(--radius-lg);
                overflow: hidden;
            }

            .panel-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                padding: 18px 20px;
                border-bottom: 1px solid var(--line);
                background: rgba(15, 23, 42, 0.5);
            }

            .panel-title {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .panel-title h3 {
                margin: 0;
                font-size: 17px;
                letter-spacing: -0.02em;
            }

            .panel-title p {
                margin: 3px 0 0;
                color: var(--muted);
                font-size: 12px;
            }

            .panel-body { padding: 20px; }

            .camera-frame {
                position: relative;
                overflow: hidden;
                border-radius: 18px;
                border: 1px solid rgba(56, 189, 248, 0.26);
                background: #030712;
                min-height: 360px;
            }

            .camera-frame img {
                width: 100%;
                display: block;
                min-height: 360px;
                object-fit: cover;
            }

            .scan-overlay {
                position: absolute;
                inset: 0;
                pointer-events: none;
                background:
                    linear-gradient(transparent 0%, rgba(56, 189, 248, 0.06) 50%, transparent 100%),
                    repeating-linear-gradient(0deg, rgba(255,255,255,0.03) 0 1px, transparent 1px 5px);
                mix-blend-mode: screen;
            }

            .heatmap-canvas {
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                opacity: 0.52;
                mix-blend-mode: screen;
                transition: opacity 0.25s ease;
            }

            .heatmap-canvas.off {
                opacity: 0;
            }

            .heat-toggle-chip {
                cursor: pointer;
                pointer-events: auto;
                user-select: none;
                border-color: rgba(56, 189, 248, 0.34);
                background: rgba(15, 23, 42, 0.82);
                transition: border-color 0.2s ease, background 0.2s ease, transform 0.2s ease;
            }

            .heat-toggle-chip:hover {
                transform: translateY(-1px);
                border-color: rgba(56, 189, 248, 0.72);
            }

            .heat-toggle-chip strong {
                color: #94a3b8;
                letter-spacing: 0.02em;
            }

            .heat-toggle-chip.active {
                background: rgba(245, 158, 11, 0.16);
                border-color: rgba(245, 158, 11, 0.68);
                box-shadow: 0 0 22px rgba(245, 158, 11, 0.18);
            }

            .heat-toggle-chip.active strong {
                color: #fde68a;
            }

            .heat-toggle-chip .mini-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #64748b;
                box-shadow: none;
            }

            .heat-toggle-chip.active .mini-dot {
                background: #f59e0b;
                box-shadow: 0 0 12px rgba(245, 158, 11, 0.95);
            }

            .heatmap-action-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                margin-top: 12px;
                padding: 12px 14px;
                border-radius: 16px;
                background: rgba(2, 6, 23, 0.56);
                border: 1px solid rgba(148, 163, 184, 0.16);
            }

            .heatmap-action-row p {
                margin: 0;
                color: var(--muted);
                font-size: 12px;
                line-height: 1.45;
            }

            .heatmap-action-row strong {
                display: block;
                color: var(--text);
                margin-bottom: 3px;
                font-size: 13px;
                letter-spacing: 0.02em;
            }

            .heatmap-button {
                border: none;
                border-radius: 999px;
                padding: 11px 15px;
                cursor: pointer;
                color: #e2e8f0;
                font-weight: 950;
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.96));
                border: 1px solid rgba(56, 189, 248, 0.34);
                white-space: nowrap;
            }

            .heatmap-button.active {
                color: #fff7ed;
                background: linear-gradient(135deg, #f59e0b, #dc2626);
                border-color: rgba(251, 191, 36, 0.74);
                box-shadow: 0 12px 32px rgba(220, 38, 38, 0.22);
            }

            .heatmap-ops-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
                margin-top: 14px;
            }

            .heatmap-card {
                position: relative;
                min-height: 98px;
                padding: 14px;
                overflow: hidden;
                border-radius: 16px;
                background: rgba(15, 23, 42, 0.74);
                border: 1px solid var(--line);
            }

            .heatmap-card::after {
                content: "";
                position: absolute;
                inset: auto -20px -36px auto;
                width: 116px;
                height: 116px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(245, 158, 11, 0.22), transparent 68%);
            }

            .heatmap-card span {
                display: block;
                color: var(--muted);
                font-size: 11px;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .heatmap-card strong {
                position: relative;
                z-index: 2;
                display: block;
                margin-top: 8px;
                font-size: 20px;
                letter-spacing: -0.03em;
            }

            .heatmap-card small {
                position: relative;
                z-index: 2;
                display: block;
                margin-top: 7px;
                color: var(--muted-2);
                line-height: 1.38;
            }

            .heat-legend {
                display: flex;
                gap: 8px;
                align-items: center;
                flex-wrap: wrap;
                margin-top: 10px;
                color: var(--muted);
                font-size: 11px;
                font-weight: 800;
            }

            .heat-swatch {
                width: 11px;
                height: 11px;
                border-radius: 50%;
                display: inline-block;
                box-shadow: 0 0 12px rgba(255,255,255,0.16);
            }

            .heat-swatch.low { background: #22c55e; }
            .heat-swatch.medium { background: #facc15; }
            .heat-swatch.high { background: #f97316; }
            .heat-swatch.critical { background: #ef4444; }

            .camera-hud {
                position: absolute;
                left: 14px;
                right: 14px;
                top: 14px;
                display: flex;
                justify-content: space-between;
                gap: 10px;
                flex-wrap: wrap;
            }

            .hud-chip {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 11px;
                border-radius: 999px;
                background: rgba(2, 6, 23, 0.74);
                border: 1px solid rgba(148, 163, 184, 0.2);
                color: #e2e8f0;
                font-size: 12px;
                font-weight: 800;
                backdrop-filter: blur(10px);
            }

            .signal-stack {
                position: absolute;
                right: 16px;
                bottom: 16px;
                width: 58px;
                padding: 10px;
                border-radius: 22px;
                background: rgba(2, 6, 23, 0.78);
                border: 1px solid rgba(148, 163, 184, 0.22);
                backdrop-filter: blur(10px);
                display: grid;
                gap: 8px;
            }

            .signal-dot {
                width: 38px;
                height: 38px;
                border-radius: 50%;
                background: rgba(100, 116, 139, 0.34);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }

            .signal-dot.red.active { background: var(--red); box-shadow: 0 0 22px rgba(239, 68, 68, 0.7); }
            .signal-dot.amber.active { background: var(--amber); box-shadow: 0 0 22px rgba(245, 158, 11, 0.7); }
            .signal-dot.green.active { background: var(--green); box-shadow: 0 0 22px rgba(34, 197, 94, 0.72); }

            .status-row {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
                margin-top: 14px;
            }

            .status-tile {
                padding: 14px;
                border-radius: 14px;
                background: rgba(15, 23, 42, 0.72);
                border: 1px solid var(--line);
            }

            .status-tile span {
                display: block;
                color: var(--muted);
                font-size: 11px;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .status-tile strong {
                display: block;
                margin-top: 8px;
                font-size: 15px;
                line-height: 1.3;
            }

            .two-col {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 18px;
            }

            canvas {
                width: 100%;
                height: 220px;
                border-radius: 16px;
                background: rgba(2, 6, 23, 0.72);
                border: 1px solid var(--line);
            }

            .ai-decision {
                border-radius: 18px;
                padding: 18px;
                background:
                    linear-gradient(135deg, rgba(8, 47, 73, 0.82), rgba(19, 78, 74, 0.74)),
                    rgba(2, 6, 23, 0.72);
                border: 1px solid rgba(94, 234, 212, 0.30);
                position: relative;
                overflow: hidden;
            }

            .ai-decision::before {
                content: "AI";
                position: absolute;
                right: 16px;
                top: 12px;
                color: rgba(255, 255, 255, 0.07);
                font-size: 64px;
                font-weight: 1000;
                letter-spacing: -0.08em;
            }

            .ai-decision h4 {
                margin: 0 0 9px;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: #99f6e4;
            }

            .ai-decision p {
                position: relative;
                margin: 0;
                color: #e0f2fe;
                line-height: 1.62;
                font-size: 14px;
            }

            .insight-list {
                display: grid;
                gap: 10px;
                margin-top: 12px;
            }

            .insight-item {
                display: grid;
                grid-template-columns: 28px 1fr;
                gap: 10px;
                align-items: start;
                padding: 12px;
                border-radius: 14px;
                background: rgba(15, 23, 42, 0.7);
                border: 1px solid var(--line);
                color: #dbeafe;
                font-size: 13px;
                line-height: 1.45;
            }

            .insight-item .badge-icon {
                width: 28px;
                height: 28px;
                border-radius: 10px;
                display: grid;
                place-items: center;
                background: var(--purple-soft);
                color: var(--purple);
            }

            .control-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
            }

            .control-card {
                padding: 14px;
                border-radius: 16px;
                border: 1px solid var(--line);
                background: rgba(15, 23, 42, 0.72);
            }

            .control-card h4 {
                margin: 0 0 6px;
                font-size: 14px;
            }

            .control-card p {
                min-height: 38px;
                margin: 0 0 12px;
                color: var(--muted);
                font-size: 12px;
                line-height: 1.45;
            }

            .button-row {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }

            button {
                border: 0;
                outline: 0;
                color: white;
                font-weight: 900;
                letter-spacing: -0.01em;
                border-radius: 12px;
                padding: 10px 12px;
                cursor: pointer;
                transition: transform 0.15s ease, filter 0.15s ease, opacity 0.15s ease;
            }

            button:hover { transform: translateY(-1px); filter: brightness(1.08); }
            button:active { transform: translateY(0); opacity: 0.86; }

            .btn-danger { background: linear-gradient(135deg, #ef4444, #991b1b); }
            .btn-safe { background: linear-gradient(135deg, #22c55e, #166534); }
            .btn-neutral { background: linear-gradient(135deg, #475569, #1e293b); }

            .alert-box {
                display: none;
                margin-top: 12px;
                padding: 12px;
                border-radius: 14px;
                font-size: 13px;
                line-height: 1.45;
            }

            .alert-box.emergency {
                background: var(--red-soft);
                border: 1px solid rgba(248, 113, 113, 0.34);
                color: #fecaca;
            }

            .alert-box.incident {
                background: var(--amber-soft);
                border: 1px solid rgba(251, 191, 36, 0.34);
                color: #fde68a;
            }

            .vehicle-grid {
                display: grid;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                gap: 10px;
            }

            .vehicle-card {
                min-height: 92px;
                padding: 14px;
                border-radius: 16px;
                background: rgba(15, 23, 42, 0.72);
                border: 1px solid var(--line);
            }

            .vehicle-card span {
                color: var(--muted);
                font-size: 12px;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.07em;
            }

            .vehicle-card strong {
                display: block;
                margin: 8px 0 10px;
                font-size: 26px;
                letter-spacing: -0.05em;
            }

            .bar {
                width: 100%;
                height: 6px;
                overflow: hidden;
                border-radius: 999px;
                background: rgba(100, 116, 139, 0.28);
            }

            .bar i {
                display: block;
                height: 100%;
                width: 0%;
                border-radius: inherit;
                background: linear-gradient(90deg, var(--cyan), var(--green));
                transition: width 0.3s ease;
            }

            .plates-list {
                display: grid;
                gap: 8px;
                max-height: 250px;
                overflow: auto;
                padding-right: 4px;
            }

            .plate {
                display: flex;
                justify-content: space-between;
                gap: 10px;
                align-items: center;
                padding: 10px 12px;
                border-radius: 13px;
                background: rgba(2, 6, 23, 0.62);
                border: 1px solid var(--line);
                font-weight: 900;
                letter-spacing: 0.08em;
            }

            .plate small {
                color: var(--muted-2);
                font-weight: 800;
                letter-spacing: normal;
            }

            .timeline {
                display: grid;
                gap: 10px;
                max-height: 280px;
                overflow: auto;
                padding-right: 4px;
            }

            .timeline-item {
                display: grid;
                grid-template-columns: 72px 1fr;
                gap: 10px;
                padding: 11px;
                border-radius: 14px;
                border: 1px solid var(--line);
                background: rgba(15, 23, 42, 0.68);
            }

            .timeline-time {
                color: var(--cyan);
                font-size: 12px;
                font-weight: 900;
            }

            .timeline-text {
                color: #cbd5e1;
                font-size: 13px;
                line-height: 1.42;
            }

            .health-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
            }

            .health-tile {
                padding: 13px;
                border-radius: 14px;
                background: rgba(15, 23, 42, 0.7);
                border: 1px solid var(--line);
            }

            .health-tile span {
                display: block;
                color: var(--muted);
                font-size: 11px;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .health-tile strong {
                display: block;
                margin-top: 7px;
                font-size: 18px;
            }

            .camera-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
            }

            .mini-camera {
                min-height: 132px;
                position: relative;
                overflow: hidden;
                border-radius: 16px;
                border: 1px solid var(--line);
                background:
                    linear-gradient(135deg, rgba(15, 23, 42, 0.74), rgba(8, 47, 73, 0.5)),
                    radial-gradient(circle at 30% 30%, rgba(56, 189, 248, 0.18), transparent 32%);
                padding: 14px;
            }

            .mini-camera::before {
                content: "";
                position: absolute;
                inset: 48px 16px auto;
                height: 1px;
                background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.8), transparent);
            }

            .mini-camera strong {
                display: block;
                font-size: 14px;
                margin-bottom: 6px;
            }

            .mini-camera span {
                color: var(--muted);
                font-size: 12px;
                line-height: 1.4;
            }

            .status-badge {
                display: inline-flex;
                align-items: center;
                gap: 7px;
                border-radius: 999px;
                padding: 6px 9px;
                border: 1px solid var(--line);
                color: #cbd5e1;
                font-size: 11px;
                font-weight: 900;
                background: rgba(2, 6, 23, 0.56);
            }

            .status-badge.safe { color: #bbf7d0; border-color: rgba(34, 197, 94, 0.28); background: rgba(34, 197, 94, 0.10); }
            .status-badge.warn { color: #fde68a; border-color: rgba(245, 158, 11, 0.32); background: rgba(245, 158, 11, 0.11); }
            .status-badge.danger { color: #fecaca; border-color: rgba(239, 68, 68, 0.32); background: rgba(239, 68, 68, 0.12); }

            .metric-good { color: var(--green); }
            .metric-warn { color: var(--amber); }
            .metric-danger { color: var(--red); }
            .metric-cyan { color: var(--cyan); }

            .progress-ring {
                width: 56px;
                height: 56px;
                border-radius: 50%;
                display: grid;
                place-items: center;
                background: conic-gradient(var(--cyan) 0deg, rgba(100, 116, 139, 0.25) 0deg);
                position: relative;
            }

            .progress-ring::after {
                content: "";
                position: absolute;
                inset: 6px;
                border-radius: 50%;
                background: #0b1220;
            }

            .progress-ring strong {
                position: relative;
                z-index: 2;
                font-size: 12px;
            }


            .digital-twin-grid {
                display: grid;
                grid-template-columns: minmax(0, 1.45fr) minmax(300px, 0.72fr);
                gap: 18px;
                align-items: stretch;
            }

            .twin-stage {
                position: relative;
                min-height: 470px;
                overflow: hidden;
                border-radius: 20px;
                border: 1px solid rgba(56, 189, 248, 0.28);
                background:
                    radial-gradient(circle at 50% 50%, rgba(56, 189, 248, 0.14), transparent 32%),
                    linear-gradient(135deg, rgba(2, 6, 23, 0.92), rgba(8, 47, 73, 0.45));
            }

            .twin-stage canvas {
                width: 100%;
                height: 470px;
                display: block;
                border: 0;
                border-radius: 0;
                background: transparent;
            }

            .twin-hud {
                position: absolute;
                inset: 14px 14px auto 14px;
                display: flex;
                gap: 8px;
                justify-content: space-between;
                flex-wrap: wrap;
                pointer-events: none;
            }

            .twin-chip {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 10px;
                border-radius: 999px;
                background: rgba(2, 6, 23, 0.72);
                border: 1px solid rgba(148, 163, 184, 0.22);
                color: #e2e8f0;
                font-size: 11px;
                font-weight: 900;
                backdrop-filter: blur(10px);
            }

            .twin-legend {
                position: absolute;
                left: 14px;
                bottom: 14px;
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                pointer-events: none;
            }

            .legend-dot {
                width: 9px;
                height: 9px;
                border-radius: 50%;
                display: inline-block;
            }

            .legend-dot.car { background: #38bdf8; }
            .legend-dot.bus { background: #f59e0b; }
            .legend-dot.truck { background: #a78bfa; }
            .legend-dot.priority { background: #22c55e; box-shadow: 0 0 12px rgba(34, 197, 94, 0.9); }

            .twin-console {
                display: grid;
                gap: 12px;
            }

            .twin-console-card {
                padding: 14px;
                border-radius: 16px;
                background: rgba(15, 23, 42, 0.72);
                border: 1px solid var(--line);
            }

            .twin-console-card h4 {
                margin: 0 0 10px;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #cbd5e1;
            }

            .phase-row {
                display: grid;
                grid-template-columns: 1fr auto;
                gap: 10px;
                align-items: center;
                padding: 9px 0;
                border-bottom: 1px solid rgba(148, 163, 184, 0.12);
                color: var(--muted);
                font-size: 13px;
            }

            .phase-row:last-child { border-bottom: 0; }

            .phase-row strong {
                color: #e2e8f0;
                font-size: 13px;
            }

            .lane-bars {
                display: grid;
                gap: 10px;
            }

            .lane-meter {
                display: grid;
                grid-template-columns: 82px 1fr 38px;
                align-items: center;
                gap: 10px;
                font-size: 12px;
                color: var(--muted);
            }

            .lane-meter strong { color: #e2e8f0; }

            .lane-track {
                height: 8px;
                border-radius: 999px;
                overflow: hidden;
                background: rgba(100, 116, 139, 0.24);
            }

            .lane-track i {
                display: block;
                width: 0%;
                height: 100%;
                border-radius: inherit;
                background: linear-gradient(90deg, var(--green), var(--cyan), var(--amber));
                transition: width 0.32s ease;
            }

            .signal-plan {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 8px;
            }

            .signal-plan span {
                padding: 10px 8px;
                border-radius: 13px;
                background: rgba(2, 6, 23, 0.58);
                border: 1px solid var(--line);
                color: var(--muted);
                font-size: 11px;
                text-align: center;
                font-weight: 900;
            }

            .signal-plan span.active {
                color: #bbf7d0;
                border-color: rgba(34, 197, 94, 0.35);
                background: rgba(34, 197, 94, 0.12);
                box-shadow: inset 0 0 20px rgba(34, 197, 94, 0.08);
            }

            .twin-note {
                margin: 10px 0 0;
                color: var(--muted);
                font-size: 12px;
                line-height: 1.5;
            }

            .footer {
                margin: 18px 0 4px;
                padding: 16px 18px;
                border-radius: 18px;
                border: 1px solid var(--line);
                background: rgba(2, 6, 23, 0.62);
                color: var(--muted);
                display: flex;
                justify-content: space-between;
                gap: 12px;
                flex-wrap: wrap;
                font-size: 12px;
            }

            @media (max-width: 1260px) {
                .app-shell { grid-template-columns: 90px minmax(0, 1fr); }
                .brand { justify-content: center; padding: 10px 0 20px; }
                .brand div:not(.brand-mark), .nav-item span:not(.nav-icon), .nav-label, .sidebar-footer { display: none; }
                .nav-item { justify-content: center; }
                .sidebar { padding: 22px 12px; }
            }

            @media (max-width: 1100px) {
                .layout-grid, .two-col, .digital-twin-grid { grid-template-columns: 1fr; }
                .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                .vehicle-grid, .camera-grid, .heatmap-ops-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                .status-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }

            @media (max-width: 760px) {
                .app-shell { display: block; }
                .sidebar { position: relative; width: auto; height: auto; display: flex; overflow-x: auto; gap: 8px; border-right: 0; border-bottom: 1px solid var(--line); }
                .brand { min-width: 58px; border-bottom: 0; margin: 0; padding: 0; }
                .nav-item { min-width: 48px; margin-bottom: 0; }
                .main { padding: 16px; }
                .topbar { align-items: flex-start; flex-direction: column; }
                .ops-strip { justify-content: flex-start; }
                .kpi-grid, .status-row, .vehicle-grid, .control-grid, .health-grid, .camera-grid, .heatmap-ops-grid, .signal-plan { grid-template-columns: 1fr; }
                .camera-frame, .camera-frame img { min-height: 260px; }
                .twin-stage { min-height: 360px; }
                .twin-stage canvas { height: 360px; }
            }
        </style>
    </head>

    <body>
        <div class="app-shell">
            <aside class="sidebar">
                <div class="brand">
                    <div class="brand-mark">VF</div>
                    <div>
                        <h1>VisionFlow AI</h1>
                        <p>Smart Mobility Operations Center</p>
                    </div>
                </div>

                <div class="nav-label">Operations</div>
                <a class="nav-item active" href="#overview"><span class="nav-icon">⌁</span><span>Overview</span></a>
                <a class="nav-item" href="#live-feed"><span class="nav-icon">◉</span><span>Live CV Feed</span></a>
                <a class="nav-item" href="#heatmap-panel"><span class="nav-icon">◌</span><span>Heat Map</span></a>
                <a class="nav-item" href="#digital-twin"><span class="nav-icon">⌬</span><span>Digital Twin</span></a>
                <a class="nav-item" href="#signals"><span class="nav-icon">⬡</span><span>Signal AI</span></a>
                <a class="nav-item" href="#controls"><span class="nav-icon">⚑</span><span>Safety Controls</span></a>
                <a class="nav-item" href="#analytics"><span class="nav-icon">▣</span><span>Traffic Analytics</span></a>
                <a class="nav-item" href="#system"><span class="nav-icon">⚙</span><span>System Health</span></a>

                <div class="sidebar-footer">
                    <strong>FORVIA Demo Mode</strong>
                    <span>Computer Vision + adaptive signal intelligence + explainable AI decision layer.</span>
                </div>
            </aside>

            <main class="main" id="overview">
                <section class="topbar">
                    <div class="hero">
                        <h2>AI-Powered Intelligent Traffic Management Platform</h2>
                        <p>
                            Enterprise-grade Smart Mobility dashboard for real-time vehicle classification, congestion intelligence,
                            adaptive traffic signals, OCR-based plate logging, emergency priority, and incident-response workflows.
                        </p>
                    </div>
                    <div class="ops-strip">
                        <span class="pill"><span class="pulse-dot"></span> Live CV Pipeline</span>
                        <span class="pill" id="clockPill">--:--:-- IST</span>
                        <span class="pill">Intersection A-01</span>
                    </div>
                </section>

                <section class="kpi-grid">
                    <article class="kpi-card">
                        <div class="kpi-head">
                            <span class="kpi-label">Detected Vehicles</span>
                            <span class="kpi-icon">▦</span>
                        </div>
                        <div class="kpi-value metric-cyan" id="vehicleCount">0</div>
                        <div class="kpi-sub">YOLOv8 classified vehicles in the active video window.</div>
                    </article>

                    <article class="kpi-card">
                        <div class="kpi-head">
                            <span class="kpi-label">Signal Timing</span>
                            <span class="kpi-icon">◷</span>
                        </div>
                        <div class="kpi-value metric-good" id="waitTime">0s</div>
                        <div class="kpi-sub">Recommended adaptive green phase duration.</div>
                    </article>

                    <article class="kpi-card">
                        <div class="kpi-head">
                            <span class="kpi-label">Congestion Index</span>
                            <span class="kpi-icon">≋</span>
                        </div>
                        <div class="kpi-value" id="congestionScore">0%</div>
                        <div class="kpi-sub" id="congestionText">Low traffic density detected.</div>
                    </article>

                    <article class="kpi-card">
                        <div class="kpi-head">
                            <span class="kpi-label">AI Confidence</span>
                            <div class="progress-ring" id="confidenceRing"><strong id="aiConfidence">0%</strong></div>
                        </div>
                        <div class="kpi-value" id="safetyStatus">Safe</div>
                        <div class="kpi-sub">Safety and decision reliability status.</div>
                    </article>
                </section>

                <section class="layout-grid">
                    <div class="left-stack">
                        <section class="panel" id="live-feed">
                            <div class="panel-header">
                                <div class="panel-title">
                                    <span class="nav-icon">◉</span>
                                    <div>
                                        <h3>Live Computer Vision Feed</h3>
                                        <p>YOLOv8 vehicle detection, classification overlay, OCR plate feed, and operator HUD.</p>
                                    </div>
                                </div>
                                <span class="status-badge safe" id="feedBadge"><span class="pulse-dot"></span>ONLINE</span>
                            </div>
                            <div class="panel-body">
                                <div class="camera-frame">
                                    <img src="/video_feed" alt="Live traffic video feed" />
                                    <canvas id="liveHeatmapCanvas" class="heatmap-canvas off" width="640" height="360"></canvas>
                                    <div class="scan-overlay"></div>
                                    <div class="camera-hud">
                                        <span class="hud-chip">CAM-A01 · Primary Junction</span>
                                        <span class="hud-chip" id="hudSignal">Signal: Normal</span>
                                        <span class="hud-chip" id="hudMode">AI Monitoring Active</span>
                                        <span class="hud-chip heat-toggle-chip" id="heatmapHudButton" onclick="toggleHeatMap()"><span class="mini-dot"></span> Real-Time Heat: <strong id="heatmapToggleLabel">OFF</strong></span>
                                    </div>
                                    <div class="signal-stack" aria-label="Traffic signal state">
                                        <div class="signal-dot red" id="signalRed"></div>
                                        <div class="signal-dot amber" id="signalAmber"></div>
                                        <div class="signal-dot green active" id="signalGreen"></div>
                                    </div>
                                </div>

                                <div class="status-row">
                                    <div class="status-tile"><span>Safety Status</span><strong id="safetyTile">Safe</strong></div>
                                    <div class="status-tile"><span>Signal Mode</span><strong id="signalMode">Normal</strong></div>
                                    <div class="status-tile"><span>System Mode</span><strong id="systemMode">AI Monitoring Active</strong></div>
                                    <div class="status-tile"><span>Prediction</span><strong><span id="prediction">0</span> · <span id="trend">Stable</span></strong></div>
                                </div>

                                <div class="heatmap-ops-grid" id="heatmap-panel">
                                    <div class="heatmap-card"><span>Heat Map Overlay</span><strong id="heatmapStatus">Overlay OFF</strong><small id="heatmapStatusSub">Camera view stays clean. Turn on heat only when congestion needs density inspection.</small></div>
                                    <div class="heatmap-card"><span>Peak Density Zone</span><strong id="heatmapPeakZone">Center Junction</strong><small>Computed from vehicle mix and lane-pressure estimate.</small></div>
                                    <div class="heatmap-card"><span>Heavy-Vehicle Heat</span><strong id="heatmapHeavy">0%</strong><small>Bus/truck contribution to weighted density.</small></div>
                                    <div class="heatmap-card"><span>Overlay Coverage</span><strong id="heatmapCoverage">0%</strong><small>Estimated road-surface heat coverage from detections.</small></div>
                                </div>
                                <div class="heatmap-action-row">
                                    <p><strong>Real-Time Density Overlay</strong><span id="heatmapModeHint">OFF by default so the CCTV feed remains clear. Click ON during high congestion to inspect hot zones.</span></p>
                                    <button class="heatmap-button" id="heatmapActionButton" onclick="toggleHeatMap()">Turn Heat Map ON</button>
                                </div>
                                <div class="heat-legend">
                                    <span><i class="heat-swatch low"></i> Low</span>
                                    <span><i class="heat-swatch medium"></i> Medium</span>
                                    <span><i class="heat-swatch high"></i> High</span>
                                    <span><i class="heat-swatch critical"></i> Critical</span>
                                </div>
                            </div>
                        </section>


                        <section class="panel" id="digital-twin">
                            <div class="panel-header">
                                <div class="panel-title">
                                    <span class="nav-icon">⌬</span>
                                    <div>
                                        <h3>Live Intersection Digital Twin</h3>
                                        <p>2D operational twin showing signal phase, vehicle movement paths, lane density, and control-room state from live AI data.</p>
                                    </div>
                                </div>
                                <span class="status-badge safe" id="twinStatusBadge"><span class="pulse-dot"></span>SYNCED</span>
                            </div>
                            <div class="panel-body">
                                <div class="digital-twin-grid">
                                    <div class="twin-stage">
                                        <canvas id="digitalTwinCanvas" width="920" height="520"></canvas>
                                        <div class="twin-hud">
                                            <span class="twin-chip" id="twinModeChip">Mode: AI Monitoring Active</span>
                                            <span class="twin-chip" id="twinSignalChip">Signal: Normal</span>
                                            <span class="twin-chip" id="twinFlowChip">Flow: Smooth</span>
                                            <span class="twin-chip" id="twinTimerChip">Green Phase: 25s</span>
                                        </div>
                                        <div class="twin-legend">
                                            <span class="twin-chip"><i class="legend-dot car"></i>Cars/Bikes</span>
                                            <span class="twin-chip"><i class="legend-dot bus"></i>Bus</span>
                                            <span class="twin-chip"><i class="legend-dot truck"></i>Truck</span>
                                            <span class="twin-chip"><i class="legend-dot priority"></i>Priority Corridor</span>
                                        </div>
                                    </div>

                                    <div class="twin-console">
                                        <div class="twin-console-card">
                                            <h4>Signal Phase Plan</h4>
                                            <div class="signal-plan">
                                                <span id="phaseNormal" class="active">Adaptive</span>
                                                <span id="phasePriority">Priority</span>
                                                <span id="phaseIncident">Incident</span>
                                            </div>
                                            <p class="twin-note" id="twinPlanText">Normal AI control. Optimizing green phase using congestion score and predicted vehicle count.</p>
                                        </div>

                                        <div class="twin-console-card">
                                            <h4>Lane Density</h4>
                                            <div class="lane-bars">
                                                <div class="lane-meter"><strong>North</strong><div class="lane-track"><i id="laneNorth"></i></div><span id="laneNorthVal">0%</span></div>
                                                <div class="lane-meter"><strong>East</strong><div class="lane-track"><i id="laneEast"></i></div><span id="laneEastVal">0%</span></div>
                                                <div class="lane-meter"><strong>South</strong><div class="lane-track"><i id="laneSouth"></i></div><span id="laneSouthVal">0%</span></div>
                                                <div class="lane-meter"><strong>West</strong><div class="lane-track"><i id="laneWest"></i></div><span id="laneWestVal">0%</span></div>
                                            </div>
                                        </div>

                                        <div class="twin-console-card">
                                            <h4>Digital Twin State</h4>
                                            <div class="phase-row"><span>Active Vehicles</span><strong id="twinVehiclesCount">0</strong></div>
                                            <div class="phase-row"><span>Predicted Queue</span><strong id="twinQueue">0 vehicles</strong></div>
                                            <div class="phase-row"><span>Congestion Heat</span><strong id="twinHeat">Low</strong></div>
                                            <div class="phase-row"><span>Movement Strategy</span><strong id="twinStrategy">Balanced Flow</strong></div>
                                            <p class="twin-note">This twin currently simulates movement from live counts. In the next stage, it can be mapped to object tracking IDs and exact lane coordinates.</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </section>

                        <section class="two-col" id="signals">
                            <div class="panel">
                                <div class="panel-header">
                                    <div class="panel-title">
                                        <span class="nav-icon">⬡</span>
                                        <div>
                                            <h3>Traffic Trend Intelligence</h3>
                                            <p>Live vehicle trend, traffic movement, and next-window prediction.</p>
                                        </div>
                                    </div>
                                </div>
                                <div class="panel-body">
                                    <canvas id="trendChart" width="760" height="220"></canvas>
                                </div>
                            </div>

                            <div class="panel">
                                <div class="panel-header">
                                    <div class="panel-title">
                                        <span class="nav-icon">✦</span>
                                        <div>
                                            <h3>Explainable AI Decision Engine</h3>
                                            <p>Reasoning layer for signal control and operator action.</p>
                                        </div>
                                    </div>
                                </div>
                                <div class="panel-body">
                                    <div class="ai-decision">
                                        <h4>Recommendation</h4>
                                        <p id="aiRecommendation">Waiting for traffic analysis...</p>
                                    </div>
                                    <div class="insight-list" id="insightList"></div>
                                </div>
                            </div>
                        </section>

                        <section class="panel" id="analytics">
                            <div class="panel-header">
                                <div class="panel-title">
                                    <span class="nav-icon">▣</span>
                                    <div>
                                        <h3>Vehicle Composition Analytics</h3>
                                        <p>Traffic mix for density scoring, heavy-vehicle impact, and flow analysis.</p>
                                    </div>
                                </div>
                                <span class="status-badge" id="heavyVehicleBadge">Heavy Vehicle: 0%</span>
                            </div>
                            <div class="panel-body">
                                <div class="vehicle-grid">
                                    <div class="vehicle-card"><span>Cars</span><strong id="cars">0</strong><div class="bar"><i id="barCars"></i></div></div>
                                    <div class="vehicle-card"><span>Buses</span><strong id="buses">0</strong><div class="bar"><i id="barBuses"></i></div></div>
                                    <div class="vehicle-card"><span>Trucks</span><strong id="trucks">0</strong><div class="bar"><i id="barTrucks"></i></div></div>
                                    <div class="vehicle-card"><span>Motorcycles</span><strong id="motorcycles">0</strong><div class="bar"><i id="barMotorcycles"></i></div></div>
                                    <div class="vehicle-card"><span>Bicycles</span><strong id="bicycles">0</strong><div class="bar"><i id="barBicycles"></i></div></div>
                                </div>
                            </div>
                        </section>

                        <section class="panel">
                            <div class="panel-header">
                                <div class="panel-title">
                                    <span class="nav-icon">▤</span>
                                    <div>
                                        <h3>Multi-Camera Monitoring</h3>
                                        <p>Enterprise layout prepared for multiple junction streams. Current feed powers Camera 1.</p>
                                    </div>
                                </div>
                            </div>
                            <div class="panel-body">
                                <div class="camera-grid">
                                    <div class="mini-camera"><strong>Camera 1 · Active</strong><span id="cam1Status">Primary intersection feed · live analytics attached.</span></div>
                                    <div class="mini-camera"><strong>Camera 2 · Standby</strong><span>Ready for northbound lane stream integration.</span></div>
                                    <div class="mini-camera"><strong>Camera 3 · Standby</strong><span>Ready for eastbound lane stream integration.</span></div>
                                    <div class="mini-camera"><strong>Camera 4 · Standby</strong><span>Ready for pedestrian and service-lane monitoring.</span></div>
                                </div>
                            </div>
                        </section>
                    </div>

                    <aside class="right-stack">
                        <section class="panel" id="controls">
                            <div class="panel-header">
                                <div class="panel-title">
                                    <span class="nav-icon">⚑</span>
                                    <div>
                                        <h3>Command & Safety Controls</h3>
                                        <p>Manual override layer retained for emergency and incident simulation.</p>
                                    </div>
                                </div>
                            </div>
                            <div class="panel-body">
                                <div class="control-grid">
                                    <div class="control-card">
                                        <h4>Emergency Vehicle Priority</h4>
                                        <p>Activates green corridor and priority signal timing for ambulance, fire, or police movement.</p>
                                        <div class="button-row">
                                            <button class="btn-danger" onclick="triggerEmergency()">Trigger Emergency</button>
                                            <button class="btn-safe" onclick="clearEmergency()">Clear</button>
                                        </div>
                                        <div class="alert-box emergency" id="emergencyBox">No emergency vehicle detected.</div>
                                    </div>

                                    <div class="control-card">
                                        <h4>Accident / Road Blockage</h4>
                                        <p>Activates incident alerting, traffic diversion mode, and signal adjustment workflow.</p>
                                        <div class="button-row">
                                            <button class="btn-danger" onclick="triggerAccident()">Trigger Incident</button>
                                            <button class="btn-safe" onclick="clearAccident()">Clear</button>
                                        </div>
                                        <div class="alert-box incident" id="accidentBox">No accident or road blockage detected.</div>
                                    </div>
                                </div>
                            </div>
                        </section>

                        <section class="panel">
                            <div class="panel-header">
                                <div class="panel-title">
                                    <span class="nav-icon">▥</span>
                                    <div>
                                        <h3>OCR Plate Intelligence</h3>
                                        <p>Latest detected plates and text candidates from EasyOCR.</p>
                                    </div>
                                </div>
                                <button class="btn-neutral" onclick="clearPlates()">Clear OCR</button>
                            </div>
                            <div class="panel-body">
                                <div class="plates-list" id="plates">
                                    <div class="plate"><span>No OCR plates yet</span><small>Waiting</small></div>
                                </div>
                            </div>
                        </section>

                        <section class="panel">
                            <div class="panel-header">
                                <div class="panel-title">
                                    <span class="nav-icon">◎</span>
                                    <div>
                                        <h3>Event Timeline</h3>
                                        <p>Operator-facing sequence of changes and AI decisions.</p>
                                    </div>
                                </div>
                            </div>
                            <div class="panel-body">
                                <div class="timeline" id="timeline"></div>
                            </div>
                        </section>

                        <section class="panel" id="system">
                            <div class="panel-header">
                                <div class="panel-title">
                                    <span class="nav-icon">⚙</span>
                                    <div>
                                        <h3>Camera & AI Health</h3>
                                        <p>Operational status derived from API polling and model workflow state.</p>
                                    </div>
                                </div>
                            </div>
                            <div class="panel-body">
                                <div class="health-grid">
                                    <div class="health-tile"><span>API Latency</span><strong id="apiLatency">-- ms</strong></div>
                                    <div class="health-tile"><span>Dashboard FPS</span><strong id="dashboardFps">--</strong></div>
                                    <div class="health-tile"><span>Model Status</span><strong>YOLOv8 Active</strong></div>
                                    <div class="health-tile"><span>OCR Engine</span><strong>EasyOCR Ready</strong></div>
                                    <div class="health-tile"><span>Camera Status</span><strong id="cameraStatus">Online</strong></div>
                                    <div class="health-tile"><span>Last Sync</span><strong id="lastSync">--:--:--</strong></div>
                                </div>
                            </div>
                        </section>
                    </aside>
                </section>

                <footer class="footer">
                    <span>VisionFlow AI · Enterprise Mobility Prototype</span>
                    <span>YOLOv8 + EasyOCR + Congestion Engine + Prediction + Adaptive Signal Recommendation</span>
                </footer>
            </main>
        </div>

        <script>
            let lastState = null;
            let eventLog = [];
            let lastFrameTime = performance.now();
            let latestTwinData = null;
            let latestLaneDensity = { north: 0, east: 0, south: 0, west: 0 };
            let twinVehicles = [];
            let twinVehicleSeed = 0;
            let heatMapEnabled = false;

            if (!CanvasRenderingContext2D.prototype.roundRect) {
                CanvasRenderingContext2D.prototype.roundRect = function(x, y, width, height, radius) {
                    const r = Math.min(radius || 0, Math.abs(width) / 2, Math.abs(height) / 2);
                    this.beginPath();
                    this.moveTo(x + r, y);
                    this.arcTo(x + width, y, x + width, y + height, r);
                    this.arcTo(x + width, y + height, x, y + height, r);
                    this.arcTo(x, y + height, x, y, r);
                    this.arcTo(x, y, x + width, y, r);
                    this.closePath();
                    return this;
                };
            }

            function nowTime() {
                return new Date().toLocaleTimeString('en-IN', { hour12: false });
            }

            function setText(id, value) {
                const el = document.getElementById(id);
                if (el) el.innerText = value;
            }

            function addEvent(text) {
                const time = nowTime();
                if (eventLog.length && eventLog[0].text === text) return;
                eventLog.unshift({ time, text });
                eventLog = eventLog.slice(0, 10);
                renderTimeline();
            }

            function renderTimeline() {
                const timeline = document.getElementById('timeline');
                if (!timeline) return;

                if (!eventLog.length) {
                    timeline.innerHTML = `
                        <div class="timeline-item">
                            <div class="timeline-time">${nowTime()}</div>
                            <div class="timeline-text">System initialized. AI monitoring pipeline is active.</div>
                        </div>`;
                    return;
                }

                timeline.innerHTML = eventLog.map(item => `
                    <div class="timeline-item">
                        <div class="timeline-time">${item.time}</div>
                        <div class="timeline-text">${item.text}</div>
                    </div>
                `).join('');
            }

            function updateClock() {
                setText('clockPill', nowTime() + ' IST');
            }

            function getCongestionClass(level) {
                if (level === 'High' || level === 'Incident Alert' || level === 'Emergency Priority') return 'metric-danger';
                if (level === 'Medium') return 'metric-warn';
                return 'metric-good';
            }

            function updateSignalLights(data) {
                const red = document.getElementById('signalRed');
                const amber = document.getElementById('signalAmber');
                const green = document.getElementById('signalGreen');
                [red, amber, green].forEach(dot => dot && dot.classList.remove('active'));

                if (data.accident_mode) {
                    red.classList.add('active');
                } else if (data.congestion_level === 'Medium') {
                    amber.classList.add('active');
                } else {
                    green.classList.add('active');
                }
            }

            function drawTrendChart(history) {
                const canvas = document.getElementById('trendChart');
                const ctx = canvas.getContext('2d');
                const width = canvas.width;
                const height = canvas.height;

                ctx.clearRect(0, 0, width, height);
                ctx.fillStyle = 'rgba(2, 6, 23, 0.35)';
                ctx.fillRect(0, 0, width, height);

                ctx.strokeStyle = 'rgba(148, 163, 184, 0.18)';
                ctx.lineWidth = 1;
                for (let i = 0; i <= 4; i++) {
                    const y = 26 + i * ((height - 56) / 4);
                    ctx.beginPath();
                    ctx.moveTo(44, y);
                    ctx.lineTo(width - 18, y);
                    ctx.stroke();
                }

                ctx.fillStyle = '#8fa3bd';
                ctx.font = '12px Inter, Arial';
                ctx.fillText('Vehicle count', 16, 18);
                ctx.fillText('live window →', width - 104, height - 12);

                if (!history || history.length < 2) {
                    ctx.fillStyle = '#94a3b8';
                    ctx.font = '14px Inter, Arial';
                    ctx.fillText('Collecting traffic history...', Math.max(42, width / 2 - 88), height / 2);
                    return;
                }

                const counts = history.map(x => x.count);
                const maxCount = Math.max(...counts, 10);
                const areaLeft = 44;
                const areaRight = width - 18;
                const areaTop = 26;
                const areaBottom = height - 34;

                const gradient = ctx.createLinearGradient(0, areaTop, 0, areaBottom);
                gradient.addColorStop(0, 'rgba(56, 189, 248, 0.24)');
                gradient.addColorStop(1, 'rgba(34, 197, 94, 0.02)');

                const points = history.map((item, index) => {
                    const x = areaLeft + (index / (history.length - 1)) * (areaRight - areaLeft);
                    const y = areaBottom - (item.count / maxCount) * (areaBottom - areaTop);
                    return { x, y };
                });

                ctx.beginPath();
                points.forEach((p, index) => index === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
                ctx.lineTo(points[points.length - 1].x, areaBottom);
                ctx.lineTo(points[0].x, areaBottom);
                ctx.closePath();
                ctx.fillStyle = gradient;
                ctx.fill();

                ctx.strokeStyle = '#38bdf8';
                ctx.lineWidth = 3;
                ctx.beginPath();
                points.forEach((p, index) => index === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
                ctx.stroke();

                points.forEach((p) => {
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 4.2, 0, Math.PI * 2);
                    ctx.fillStyle = '#22c55e';
                    ctx.fill();
                });
            }

            function renderInsights(data) {
                const heavy = (data.vehicle_types.bus || 0) + (data.vehicle_types.truck || 0);
                const total = Math.max(data.vehicle_count || 0, 1);
                const heavyPct = Math.round((heavy / total) * 100);

                const insights = [];
                insights.push(`Congestion score is ${data.congestion_score}% with ${data.congestion_level.toLowerCase()} operating condition.`);
                insights.push(`Traffic trend is ${data.traffic_trend.toLowerCase()} with next-window prediction of ${data.predicted_vehicle_count} vehicles.`);

                if (heavy > 0) insights.push(`Heavy vehicle share is ${heavyPct}%, increasing weighted density impact on signal timing.`);
                else insights.push('No heavy vehicle pressure detected in the current window.');

                if (data.emergency_mode) insights.unshift('Emergency green corridor workflow is active. Maintain priority passage until cleared.');
                if (data.accident_mode) insights.unshift('Incident workflow is active. Recommend diversion, lane clearance, and control-room escalation.');

                document.getElementById('insightList').innerHTML = insights.slice(0, 4).map(item => `
                    <div class="insight-item"><span class="badge-icon">✦</span><span>${item}</span></div>
                `).join('');
            }

            function updateVehicleBars(types) {
                const values = {
                    Cars: types.car || 0,
                    Buses: types.bus || 0,
                    Trucks: types.truck || 0,
                    Motorcycles: types.motorcycle || 0,
                    Bicycles: types.bicycle || 0
                };
                const maxValue = Math.max(...Object.values(values), 1);
                document.getElementById('barCars').style.width = ((values.Cars / maxValue) * 100) + '%';
                document.getElementById('barBuses').style.width = ((values.Buses / maxValue) * 100) + '%';
                document.getElementById('barTrucks').style.width = ((values.Trucks / maxValue) * 100) + '%';
                document.getElementById('barMotorcycles').style.width = ((values.Motorcycles / maxValue) * 100) + '%';
                document.getElementById('barBicycles').style.width = ((values.Bicycles / maxValue) * 100) + '%';
            }

            function updateBadges(data) {
                const feedBadge = document.getElementById('feedBadge');
                const heavyBadge = document.getElementById('heavyVehicleBadge');
                const total = Math.max(data.vehicle_count || 0, 1);
                const heavyPct = Math.round((((data.vehicle_types.bus || 0) + (data.vehicle_types.truck || 0)) / total) * 100);

                feedBadge.className = 'status-badge safe';
                feedBadge.innerHTML = '<span class="pulse-dot"></span>ONLINE';

                heavyBadge.innerText = `Heavy Vehicle: ${heavyPct}%`;
                heavyBadge.className = heavyPct >= 35 ? 'status-badge warn' : 'status-badge safe';
            }


            function setHeatMapState(enabled) {
                heatMapEnabled = enabled;
                const canvas = document.getElementById('liveHeatmapCanvas');
                const label = document.getElementById('heatmapToggleLabel');
                const hudButton = document.getElementById('heatmapHudButton');
                const actionButton = document.getElementById('heatmapActionButton');
                const hint = document.getElementById('heatmapModeHint');

                if (canvas) {
                    canvas.classList.toggle('off', !heatMapEnabled);
                    if (!heatMapEnabled) {
                        const ctx = canvas.getContext('2d');
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                    }
                }

                if (label) label.innerText = heatMapEnabled ? 'ON' : 'OFF';
                if (hudButton) hudButton.classList.toggle('active', heatMapEnabled);
                if (actionButton) {
                    actionButton.classList.toggle('active', heatMapEnabled);
                    actionButton.innerText = heatMapEnabled ? 'Turn Heat Map OFF' : 'Turn Heat Map ON';
                }
                if (hint) {
                    hint.innerText = heatMapEnabled
                        ? 'ON: live density layer is being drawn over the CCTV feed from current AI traffic data.'
                        : 'OFF by default so the CCTV feed remains clear. Click ON during high congestion to inspect hot zones.';
                }

                if (heatMapEnabled && latestTwinData) drawLiveHeatmap(latestTwinData);
            }

            function toggleHeatMap() {
                setHeatMapState(!heatMapEnabled);
            }

            function getHeatLevel(score, accidentMode, emergencyMode) {
                if (accidentMode) return 'Critical';
                if (emergencyMode) return 'Priority';
                if (score >= 70) return 'High';
                if (score >= 35) return 'Medium';
                return 'Low';
            }

            function heatColorForScore(score, alpha) {
                if (score >= 85) return `rgba(239, 68, 68, ${alpha})`;
                if (score >= 70) return `rgba(249, 115, 22, ${alpha})`;
                if (score >= 35) return `rgba(250, 204, 21, ${alpha})`;
                return `rgba(34, 197, 94, ${alpha})`;
            }

            function vehicleHeatWeight(type) {
                if (type === 'bus') return 2.5;
                if (type === 'truck') return 2.2;
                if (type === 'car') return 1.0;
                if (type === 'motorcycle') return 0.5;
                if (type === 'bicycle') return 0.3;
                return 1.0;
            }

            function resizeOverlayCanvas(canvas) {
                const rect = canvas.getBoundingClientRect();
                const width = Math.max(320, Math.round(rect.width));
                const height = Math.max(180, Math.round(rect.height));
                if (canvas.width !== width || canvas.height !== height) {
                    canvas.width = width;
                    canvas.height = height;
                }
                return { width, height };
            }

            function drawHeatBlob(ctx, x, y, radius, score, strength) {
                const innerAlpha = clamp(0.20 + strength * 0.08 + score / 420, 0.18, 0.78);
                const midAlpha = clamp(innerAlpha * 0.58, 0.08, 0.42);
                const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
                gradient.addColorStop(0.00, heatColorForScore(score + strength * 10, innerAlpha));
                gradient.addColorStop(0.42, heatColorForScore(score, midAlpha));
                gradient.addColorStop(0.72, 'rgba(56, 189, 248, 0.08)');
                gradient.addColorStop(1.00, 'rgba(2, 6, 23, 0)');
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(x, y, radius, 0, Math.PI * 2);
                ctx.fill();
            }

            function drawFallbackHeatMap(ctx, width, height, data) {
                const score = data.congestion_score || 0;
                const centerY = height * 0.54;
                const roadHeight = height * 0.34;
                const centers = [
                    [width * 0.50, centerY, Math.max(width, height) * 0.30, score],
                    [width * 0.30, centerY, Math.max(width, height) * 0.22, score * 0.78],
                    [width * 0.70, centerY, Math.max(width, height) * 0.22, score * 0.72],
                    [width * 0.50, centerY - roadHeight * 0.72, Math.max(width, height) * 0.18, score * 0.62],
                    [width * 0.50, centerY + roadHeight * 0.72, Math.max(width, height) * 0.18, score * 0.68]
                ];
                centers.forEach(([x, y, r, s]) => drawHeatBlob(ctx, x, y, r, s, Math.max(1, data.vehicle_count || 1)));
            }

            function drawLiveHeatmap(data) {
                const canvas = document.getElementById('liveHeatmapCanvas');
                if (!canvas) return;

                const ctx = canvas.getContext('2d');
                const { width, height } = resizeOverlayCanvas(canvas);
                ctx.clearRect(0, 0, width, height);

                if (!heatMapEnabled) return;

                const vehicles = (data.detections && data.detections.vehicles) ? data.detections.vehicles : [];
                const frameSize = (data.detections && data.detections.frame_size) ? data.detections.frame_size : null;
                let sourceWidth = frameSize && frameSize.width ? frameSize.width : 640;
                let sourceHeight = frameSize && frameSize.height ? frameSize.height : 360;

                if (vehicles.length) {
                    const maxX = Math.max(...vehicles.map(v => Math.max(v.bbox[0], v.bbox[2])));
                    const maxY = Math.max(...vehicles.map(v => Math.max(v.bbox[1], v.bbox[3])));
                    sourceWidth = Math.max(sourceWidth, maxX || 640);
                    sourceHeight = Math.max(sourceHeight, maxY || 360);
                }

                ctx.save();
                ctx.globalCompositeOperation = 'lighter';

                if (!vehicles.length) {
                    drawFallbackHeatMap(ctx, width, height, data);
                } else {
                    vehicles.forEach(vehicle => {
                        const [x1, y1, x2, y2] = vehicle.bbox;
                        const cx = ((x1 + x2) / 2 / sourceWidth) * width;
                        const cy = ((y1 + y2) / 2 / sourceHeight) * height;
                        const boxW = Math.abs(x2 - x1) / sourceWidth * width;
                        const boxH = Math.abs(y2 - y1) / sourceHeight * height;
                        const weight = vehicleHeatWeight(vehicle.class);
                        const radius = clamp(Math.max(boxW, boxH) * (1.1 + weight * 0.26), 42, 170);
                        const localScore = clamp((data.congestion_score || 0) + weight * 14, 15, 100);
                        drawHeatBlob(ctx, cx, cy, radius, localScore, weight);
                    });
                }

                if (data.accident_mode) {
                    drawHeatBlob(ctx, width * 0.5, height * 0.55, Math.max(width, height) * 0.34, 95, 4);
                }

                if (data.emergency_mode) {
                    ctx.strokeStyle = 'rgba(34, 197, 94, 0.72)';
                    ctx.lineWidth = Math.max(8, height * 0.026);
                    ctx.setLineDash([18, 12]);
                    ctx.beginPath();
                    ctx.moveTo(width * 0.04, height * 0.62);
                    ctx.lineTo(width * 0.96, height * 0.62);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }

                ctx.restore();

                ctx.save();
                ctx.globalAlpha = 0.92;
                ctx.fillStyle = 'rgba(2, 6, 23, 0.72)';
                ctx.beginPath();
                ctx.roundRect(14, height - 52, 214, 36, 12);
                ctx.fill();
                ctx.fillStyle = '#e2e8f0';
                ctx.font = '800 12px Inter, Arial';
                ctx.fillText('Traffic Density Heat Map', 28, height - 30);
                ctx.fillStyle = heatColorForScore(data.congestion_score || 0, 0.95);
                ctx.beginPath(); ctx.arc(202, height - 34, 6, 0, Math.PI * 2); ctx.fill();
                ctx.restore();
            }

            function getPeakZoneFromLaneDensity(lanes, data) {
                if (data.accident_mode) return 'Incident Core';
                if (data.emergency_mode) return 'Priority Corridor';
                const entries = Object.entries(lanes || { north: 0, east: 0, south: 0, west: 0 });
                entries.sort((a, b) => b[1] - a[1]);
                const name = entries[0] ? entries[0][0] : 'center';
                return name.charAt(0).toUpperCase() + name.slice(1) + ' Approach';
            }

            function updateHeatmapPanel(data, lanes) {
                const total = Math.max(data.vehicle_count || 0, 1);
                const heavy = (data.vehicle_types.bus || 0) + (data.vehicle_types.truck || 0);
                const heavyPct = Math.round((heavy / total) * 100);
                const coverage = clamp(Math.round((data.congestion_score || 0) * 0.82 + total * 4 + heavyPct * 0.18), 4, 100);
                const heatLevel = getHeatLevel(data.congestion_score || 0, data.accident_mode, data.emergency_mode);
                const status = data.accident_mode ? 'Critical Incident Heat' : data.emergency_mode ? 'Priority Corridor Heat' : heatLevel + ' Density';

                setText('heatmapStatus', heatMapEnabled ? status : 'Overlay OFF');
                setText('heatmapStatusSub', heatMapEnabled
                    ? `${data.congestion_score}% density index · ${data.signal_mode}`
                    : `Live heat data ready · ${data.congestion_score}% density index · click ON to inspect`);
                setText('heatmapPeakZone', getPeakZoneFromLaneDensity(lanes, data));
                setText('heatmapHeavy', heavyPct + '%');
                setText('heatmapCoverage', heatMapEnabled ? coverage + '%' : 'Ready');
            }

            function drawTwinDensityZones(ctx, width, height, data, roadW) {
                const cx = width / 2;
                const cy = height / 2;
                const lane = latestLaneDensity || { north: 0, east: 0, south: 0, west: 0 };
                const zones = [
                    { name: 'N', x: cx, y: cy - roadW * 0.95, value: lane.north },
                    { name: 'E', x: cx + roadW * 1.2, y: cy, value: lane.east },
                    { name: 'S', x: cx, y: cy + roadW * 0.95, value: lane.south },
                    { name: 'W', x: cx - roadW * 1.2, y: cy, value: lane.west },
                    { name: 'CORE', x: cx, y: cy, value: data.accident_mode ? 96 : Math.max(data.congestion_score || 0, 18) }
                ];

                ctx.save();
                ctx.globalCompositeOperation = 'lighter';
                zones.forEach(zone => {
                    const radius = zone.name === 'CORE' ? roadW * 0.92 : roadW * 0.72;
                    const gradient = ctx.createRadialGradient(zone.x, zone.y, 0, zone.x, zone.y, radius);
                    gradient.addColorStop(0.00, heatColorForScore(zone.value, clamp(0.12 + zone.value / 160, 0.16, 0.76)));
                    gradient.addColorStop(0.52, heatColorForScore(zone.value, clamp(0.06 + zone.value / 300, 0.08, 0.35)));
                    gradient.addColorStop(1.00, 'rgba(2, 6, 23, 0)');
                    ctx.fillStyle = gradient;
                    ctx.beginPath();
                    ctx.arc(zone.x, zone.y, radius, 0, Math.PI * 2);
                    ctx.fill();
                });
                ctx.restore();
            }

            function clamp(value, min, max) {
                return Math.max(min, Math.min(max, value));
            }

            function getTwinVehicleColor(type, emergency) {
                if (emergency) return '#22c55e';
                if (type === 'bus') return '#f59e0b';
                if (type === 'truck') return '#a78bfa';
                if (type === 'motorcycle' || type === 'bicycle') return '#5eead4';
                return '#38bdf8';
            }

            function getTwinVehicleSize(type) {
                if (type === 'bus') return { w: 24, h: 12 };
                if (type === 'truck') return { w: 28, h: 13 };
                if (type === 'motorcycle' || type === 'bicycle') return { w: 13, h: 7 };
                return { w: 18, h: 10 };
            }

            function buildTwinTargets(data) {
                const types = data.vehicle_types || {};
                const target = [];
                const total = clamp(data.vehicle_count || 0, 0, 28);
                const order = [
                    ['car', types.car || 0],
                    ['bus', types.bus || 0],
                    ['truck', types.truck || 0],
                    ['motorcycle', types.motorcycle || 0],
                    ['bicycle', types.bicycle || 0]
                ];

                order.forEach(([type, count]) => {
                    for (let i = 0; i < Math.min(count, 10); i++) target.push(type);
                });

                while (target.length < total) target.push('car');
                return target.slice(0, 28);
            }

            function createTwinVehicle(type, data) {
                const routes = ['northSouth', 'southNorth', 'eastWest', 'westEast', 'northEast', 'westSouth'];
                const route = routes[twinVehicleSeed % routes.length];
                const laneOffset = ((twinVehicleSeed % 3) - 1) * 18;
                const isPriority = !!data.emergency_mode && twinVehicleSeed % 4 === 0;
                twinVehicleSeed += 1;

                return {
                    id: twinVehicleSeed,
                    type,
                    route,
                    laneOffset,
                    progress: Math.random(),
                    speed: 0.0018 + Math.random() * 0.0028,
                    priority: isPriority
                };
            }

            function syncTwinVehicles(data) {
                const targetTypes = buildTwinTargets(data);

                while (twinVehicles.length > targetTypes.length) twinVehicles.pop();

                targetTypes.forEach((type, index) => {
                    if (!twinVehicles[index]) {
                        twinVehicles[index] = createTwinVehicle(type, data);
                    } else {
                        twinVehicles[index].type = type;
                        twinVehicles[index].priority = !!data.emergency_mode && index % 4 === 0;
                    }
                });
            }

            function getRoutePoint(route, progress, offset, width, height) {
                const cx = width / 2;
                const cy = height / 2;
                const road = Math.min(width, height) * 0.20;
                const p = progress % 1;

                const points = {
                    northSouth: [cx - road * 0.28 + offset * 0.35, -40, cx - road * 0.28 + offset * 0.35, height + 40],
                    southNorth: [cx + road * 0.28 + offset * 0.35, height + 40, cx + road * 0.28 + offset * 0.35, -40],
                    eastWest: [width + 40, cy - road * 0.28 + offset * 0.35, -40, cy - road * 0.28 + offset * 0.35],
                    westEast: [-40, cy + road * 0.28 + offset * 0.35, width + 40, cy + road * 0.28 + offset * 0.35],
                    northEast: [cx - road * 0.45, -40, width + 40, cy - road * 0.45],
                    westSouth: [-40, cy + road * 0.45, cx + road * 0.45, height + 40]
                }[route] || [0, cy, width, cy];

                const [x1, y1, x2, y2] = points;
                return {
                    x: x1 + (x2 - x1) * p,
                    y: y1 + (y2 - y1) * p,
                    angle: Math.atan2(y2 - y1, x2 - x1)
                };
            }

            function drawTwinRoad(ctx, width, height, data) {
                const cx = width / 2;
                const cy = height / 2;
                const roadW = Math.min(width, height) * 0.34;
                const laneW = roadW / 4;
                const score = data ? data.congestion_score || 0 : 0;
                const heatColor = score >= 70 ? 'rgba(239, 68, 68, 0.22)' : score >= 35 ? 'rgba(245, 158, 11, 0.18)' : 'rgba(34, 197, 94, 0.12)';

                ctx.clearRect(0, 0, width, height);
                ctx.fillStyle = '#07111f';
                ctx.fillRect(0, 0, width, height);

                ctx.strokeStyle = 'rgba(56, 189, 248, 0.08)';
                ctx.lineWidth = 1;
                for (let x = 0; x < width; x += 36) {
                    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
                }
                for (let y = 0; y < height; y += 36) {
                    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
                }

                ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
                ctx.fillRect(cx - roadW / 2, 0, roadW, height);
                ctx.fillRect(0, cy - roadW / 2, width, roadW);

                ctx.fillStyle = heatColor;
                ctx.fillRect(cx - roadW / 2, 0, roadW, height);
                ctx.fillRect(0, cy - roadW / 2, width, roadW);

                ctx.fillStyle = 'rgba(2, 6, 23, 0.72)';
                ctx.fillRect(cx - roadW / 2, cy - roadW / 2, roadW, roadW);

                drawTwinDensityZones(ctx, width, height, data || {}, roadW);

                ctx.strokeStyle = 'rgba(226, 232, 240, 0.36)';
                ctx.setLineDash([14, 14]);
                ctx.lineWidth = 2;
                for (let i = -1; i <= 1; i += 2) {
                    ctx.beginPath(); ctx.moveTo(cx + i * laneW, 0); ctx.lineTo(cx + i * laneW, height); ctx.stroke();
                    ctx.beginPath(); ctx.moveTo(0, cy + i * laneW); ctx.lineTo(width, cy + i * laneW); ctx.stroke();
                }
                ctx.setLineDash([]);

                ctx.strokeStyle = 'rgba(248, 250, 252, 0.55)';
                ctx.lineWidth = 3;
                const cross = 54;
                [-1, 1].forEach(dir => {
                    for (let i = -2; i <= 2; i++) {
                        ctx.beginPath();
                        ctx.moveTo(cx - roadW / 2 - cross * dir, cy + i * 10);
                        ctx.lineTo(cx - roadW / 2 - (cross - 24) * dir, cy + i * 10);
                        ctx.stroke();

                        ctx.beginPath();
                        ctx.moveTo(cx + i * 10, cy - roadW / 2 - cross * dir);
                        ctx.lineTo(cx + i * 10, cy - roadW / 2 - (cross - 24) * dir);
                        ctx.stroke();
                    }
                });

                ctx.fillStyle = 'rgba(56, 189, 248, 0.10)';
                ctx.beginPath();
                ctx.arc(cx, cy, roadW * 0.42, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = 'rgba(56, 189, 248, 0.26)';
                ctx.stroke();

                const signalColor = data && data.accident_mode ? '#ef4444' : data && data.congestion_level === 'Medium' ? '#f59e0b' : '#22c55e';
                const signalPoints = [
                    [cx - roadW * 0.62, cy - roadW * 0.62],
                    [cx + roadW * 0.62, cy - roadW * 0.62],
                    [cx - roadW * 0.62, cy + roadW * 0.62],
                    [cx + roadW * 0.62, cy + roadW * 0.62]
                ];
                signalPoints.forEach(([x, y]) => {
                    ctx.fillStyle = 'rgba(2, 6, 23, 0.88)';
                    ctx.beginPath(); ctx.arc(x, y, 15, 0, Math.PI * 2); ctx.fill();
                    ctx.fillStyle = signalColor;
                    ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.fill();
                    ctx.shadowColor = signalColor;
                    ctx.shadowBlur = 14;
                    ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.fill();
                    ctx.shadowBlur = 0;
                });

                if (data && data.emergency_mode) {
                    ctx.strokeStyle = 'rgba(34, 197, 94, 0.9)';
                    ctx.lineWidth = 5;
                    ctx.setLineDash([18, 10]);
                    ctx.beginPath();
                    ctx.moveTo(0, cy + laneW * 1.3);
                    ctx.lineTo(width, cy + laneW * 1.3);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }

                ctx.fillStyle = 'rgba(226, 232, 240, 0.72)';
                ctx.font = '12px Inter, Arial';
                ctx.fillText('NORTH APPROACH', cx - 52, 28);
                ctx.fillText('SOUTH APPROACH', cx - 52, height - 18);
                ctx.fillText('WEST', 18, cy - roadW / 2 - 14);
                ctx.fillText('EAST', width - 50, cy - roadW / 2 - 14);
            }

            function drawTwinVehicle(ctx, vehicle, width, height, data) {
                const point = getRoutePoint(vehicle.route, vehicle.progress, vehicle.laneOffset, width, height);
                const size = getTwinVehicleSize(vehicle.type);
                const color = getTwinVehicleColor(vehicle.type, vehicle.priority);

                ctx.save();
                ctx.translate(point.x, point.y);
                ctx.rotate(point.angle);
                ctx.fillStyle = color;
                ctx.shadowColor = color;
                ctx.shadowBlur = vehicle.priority ? 18 : 7;
                ctx.beginPath();
                ctx.roundRect(-size.w / 2, -size.h / 2, size.w, size.h, 4);
                ctx.fill();
                ctx.shadowBlur = 0;
                ctx.fillStyle = 'rgba(2, 6, 23, 0.55)';
                ctx.fillRect(size.w * 0.05, -size.h / 2 + 2, size.w * 0.28, size.h - 4);
                ctx.restore();
            }

            function animateDigitalTwin() {
                const canvas = document.getElementById('digitalTwinCanvas');
                if (!canvas) {
                    requestAnimationFrame(animateDigitalTwin);
                    return;
                }

                const ctx = canvas.getContext('2d');
                const width = canvas.width;
                const height = canvas.height;
                const data = latestTwinData || { congestion_score: 0, congestion_level: 'Low', emergency_mode: false, accident_mode: false };
                const speedBoost = data.emergency_mode ? 1.8 : data.accident_mode ? 0.45 : data.congestion_score >= 70 ? 0.7 : 1;

                drawTwinRoad(ctx, width, height, data);

                twinVehicles.forEach(vehicle => {
                    vehicle.progress += vehicle.speed * speedBoost;
                    if (vehicle.progress > 1) vehicle.progress = 0;
                    drawTwinVehicle(ctx, vehicle, width, height, data);
                });

                requestAnimationFrame(animateDigitalTwin);
            }

            function updateDigitalTwin(data) {
                latestTwinData = data;
                syncTwinVehicles(data);

                const total = Math.max(data.vehicle_count || 0, 1);
                const heavy = (data.vehicle_types.bus || 0) + (data.vehicle_types.truck || 0);
                const heavyPct = Math.round((heavy / total) * 100);
                const base = clamp(data.congestion_score || 0, 0, 100);
                const north = clamp(Math.round(base * 0.94 + (data.vehicle_types.bus || 0) * 7), 4, 100);
                const east = clamp(Math.round(base * 0.72 + (data.vehicle_types.car || 0) * 4), 4, 100);
                const south = clamp(Math.round(base * 0.66 + (data.vehicle_types.truck || 0) * 8), 4, 100);
                const west = clamp(Math.round(base * 0.58 + (data.vehicle_types.motorcycle || 0) * 5), 4, 100);
                latestLaneDensity = { north, east, south, west };
                updateHeatmapPanel(data, latestLaneDensity);
                const strategy = data.accident_mode ? 'Diversion Control' : data.emergency_mode ? 'Green Corridor' : heavyPct >= 35 ? 'Heavy Vehicle Relief' : data.congestion_score >= 70 ? 'Queue Clearance' : 'Balanced Flow';

                setText('twinModeChip', 'Mode: ' + data.system_mode);
                setText('twinSignalChip', 'Signal: ' + data.signal_mode);
                setText('twinFlowChip', 'Flow: ' + data.congestion_level);
                setText('twinTimerChip', 'Green Phase: ' + data.wait_time + 's');
                setText('twinVehiclesCount', data.vehicle_count);
                setText('twinQueue', data.predicted_vehicle_count + ' vehicles');
                setText('twinHeat', data.congestion_level);
                setText('twinStrategy', strategy);

                [['laneNorth', north], ['laneEast', east], ['laneSouth', south], ['laneWest', west]].forEach(([id, value]) => {
                    const bar = document.getElementById(id);
                    const val = document.getElementById(id + 'Val');
                    if (bar) bar.style.width = value + '%';
                    if (val) val.innerText = value + '%';
                });

                const phaseNormal = document.getElementById('phaseNormal');
                const phasePriority = document.getElementById('phasePriority');
                const phaseIncident = document.getElementById('phaseIncident');
                [phaseNormal, phasePriority, phaseIncident].forEach(el => el && el.classList.remove('active'));
                if (data.accident_mode) phaseIncident.classList.add('active');
                else if (data.emergency_mode) phasePriority.classList.add('active');
                else phaseNormal.classList.add('active');

                const twinBadge = document.getElementById('twinStatusBadge');
                twinBadge.className = data.accident_mode ? 'status-badge danger' : data.emergency_mode || data.congestion_score >= 70 ? 'status-badge warn' : 'status-badge safe';
                twinBadge.innerHTML = '<span class="pulse-dot"></span>' + (data.accident_mode ? 'INCIDENT SYNC' : data.emergency_mode ? 'PRIORITY SYNC' : 'SYNCED');

                const planText = data.accident_mode
                    ? 'Incident mode active. Digital twin highlights restricted movement and recommends diversion workflow.'
                    : data.emergency_mode
                        ? 'Priority corridor active. Twin reserves the east-west movement path for emergency vehicle passage.'
                        : `Adaptive phase active. ${data.wait_time}s green phase selected from congestion score and ${data.traffic_trend.toLowerCase()} trend.`;
                setText('twinPlanText', planText);
            }

            function updateStateEvents(data) {
                if (!lastState) {
                    addEvent(`Initial state: ${data.system_mode}, congestion ${data.congestion_level}.`);
                    lastState = {
                        congestion_level: data.congestion_level,
                        signal_mode: data.signal_mode,
                        emergency_mode: data.emergency_mode,
                        accident_mode: data.accident_mode,
                        wait_time: data.wait_time,
                        traffic_trend: data.traffic_trend
                    };
                    return;
                }

                if (lastState.congestion_level !== data.congestion_level) addEvent(`Congestion changed from ${lastState.congestion_level} to ${data.congestion_level}.`);
                if (lastState.signal_mode !== data.signal_mode) addEvent(`Signal mode updated to ${data.signal_mode}.`);
                if (lastState.wait_time !== data.wait_time) addEvent(`Adaptive signal timing adjusted to ${data.wait_time} seconds.`);
                if (!lastState.emergency_mode && data.emergency_mode) addEvent('Emergency priority triggered. Green corridor activated.');
                if (lastState.emergency_mode && !data.emergency_mode) addEvent('Emergency priority cleared. Returning to AI control.');
                if (!lastState.accident_mode && data.accident_mode) addEvent('Incident alert triggered. Diversion workflow recommended.');
                if (lastState.accident_mode && !data.accident_mode) addEvent('Incident alert cleared. Monitoring resumed.');
                if (lastState.traffic_trend !== data.traffic_trend) addEvent(`Traffic trend changed to ${data.traffic_trend}.`);

                lastState = {
                    congestion_level: data.congestion_level,
                    signal_mode: data.signal_mode,
                    emergency_mode: data.emergency_mode,
                    accident_mode: data.accident_mode,
                    wait_time: data.wait_time,
                    traffic_trend: data.traffic_trend
                };
            }

            async function triggerEmergency() {
                await fetch('/api/trigger-emergency', { method: 'POST' });
                fetchData();
            }

            async function clearEmergency() {
                await fetch('/api/clear-emergency', { method: 'POST' });
                fetchData();
            }

            async function triggerAccident() {
                await fetch('/api/trigger-accident', { method: 'POST' });
                fetchData();
            }

            async function clearAccident() {
                await fetch('/api/clear-accident', { method: 'POST' });
                fetchData();
            }

            async function clearPlates() {
                await fetch('/api/clear-plates');
                fetchData();
            }

            async function fetchData() {
                const start = performance.now();
                try {
                    const res = await fetch('/api/traffic-data', { cache: 'no-store' });
                    const data = await res.json();
                    const latency = Math.round(performance.now() - start);
                    const now = performance.now();
                    const fps = Math.max(1, Math.round(1000 / Math.max(now - lastFrameTime, 1)));
                    lastFrameTime = now;

                    setText('vehicleCount', data.vehicle_count);
                    setText('waitTime', data.wait_time + 's');
                    setText('congestionScore', data.congestion_score + '%');
                    setText('congestionText', data.congestion_level + ' · ' + data.signal_mode);
                    setText('safetyStatus', data.safety_status);
                    setText('safetyTile', data.safety_status);
                    setText('signalMode', data.signal_mode);
                    setText('systemMode', data.system_mode);
                    setText('prediction', data.predicted_vehicle_count);
                    setText('trend', data.traffic_trend);
                    setText('aiConfidence', data.ai_confidence + '%');
                    setText('hudSignal', 'Signal: ' + data.signal_mode);
                    setText('hudMode', data.system_mode);
                    setText('aiRecommendation', data.ai_recommendation);
                    setText('apiLatency', latency + ' ms');
                    setText('dashboardFps', fps + ' fps');
                    setText('lastSync', nowTime());
                    setText('cameraStatus', 'Online');
                    setText('cam1Status', `Primary intersection · ${data.vehicle_count} vehicles · ${data.congestion_level}`);

                    const congestionEl = document.getElementById('congestionScore');
                    congestionEl.className = 'kpi-value ' + getCongestionClass(data.congestion_level);

                    const safetyEl = document.getElementById('safetyStatus');
                    safetyEl.className = 'kpi-value ' + (data.accident_mode || data.emergency_mode ? 'metric-danger' : 'metric-good');

                    const ringDegrees = Math.max(0, Math.min(100, data.ai_confidence)) * 3.6;
                    document.getElementById('confidenceRing').style.background = `conic-gradient(#38bdf8 ${ringDegrees}deg, rgba(100, 116, 139, 0.25) ${ringDegrees}deg)`;

                    const emergencyBox = document.getElementById('emergencyBox');
                    emergencyBox.innerText = data.emergency_message;
                    emergencyBox.style.display = data.emergency_mode ? 'block' : 'none';

                    const accidentBox = document.getElementById('accidentBox');
                    accidentBox.innerText = data.accident_message;
                    accidentBox.style.display = data.accident_mode ? 'block' : 'none';

                    setText('cars', data.vehicle_types.car || 0);
                    setText('buses', data.vehicle_types.bus || 0);
                    setText('trucks', data.vehicle_types.truck || 0);
                    setText('motorcycles', data.vehicle_types.motorcycle || 0);
                    setText('bicycles', data.vehicle_types.bicycle || 0);
                    updateVehicleBars(data.vehicle_types);
                    updateSignalLights(data);
                    updateBadges(data);
                    updateDigitalTwin(data);
                    drawLiveHeatmap(data);
                    renderInsights(data);
                    drawTrendChart(data.traffic_history);
                    updateStateEvents(data);

                    const platesDiv = document.getElementById('plates');
                    if (!data.plates || data.plates.length === 0) {
                        platesDiv.innerHTML = '<div class="plate"><span>No OCR plates yet</span><small>Waiting</small></div>';
                    } else {
                        platesDiv.innerHTML = data.plates.slice(-10).reverse().map((p, index) => `
                            <div class="plate"><span>${p}</span><small>${index === 0 ? 'Latest' : 'OCR'}</small></div>
                        `).join('');
                    }
                } catch (error) {
                    setText('cameraStatus', 'Offline');
                    setText('apiLatency', 'Error');
                    addEvent('API connection warning. Dashboard is waiting for Flask backend response.');
                }
            }

            updateClock();
            renderTimeline();
            animateDigitalTwin();
            window.addEventListener('resize', () => { if (latestTwinData) drawLiveHeatmap(latestTwinData); });
            setInterval(updateClock, 1000);
            setInterval(fetchData, 1000);
            setHeatMapState(false);
            fetchData();
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5000)
