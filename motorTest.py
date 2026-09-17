# Animal Detection System - High-Performance Real-Time Video
import os
import cv2
import time
import json
import threading
import queue
import pygame
from ultralytics import YOLO

# Resolve absolute paths based on this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "cowvid.mp4")
COW_SOUND_PATH = os.path.join(BASE_DIR, "animal_sounds", "cow.mp3")
ELEPHANT_SOUND_PATH = os.path.join(BASE_DIR, "animal_sounds", "beebuzz.mp3")
MODEL_PATH = os.path.join(BASE_DIR, "yolov8m.pt")

# Initialize Pygame mixer for audio deterrents
pygame.mixer.init()

cow_sound = None
elephant_sound = None
if os.path.exists(COW_SOUND_PATH):
    cow_sound = pygame.mixer.Sound(COW_SOUND_PATH)
if os.path.exists(ELEPHANT_SOUND_PATH):
    elephant_sound = pygame.mixer.Sound(ELEPHANT_SOUND_PATH)

# Load YOLO model
print("⏳ Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ YOLO model loaded")

# ----------------- Arduino / Simulation Setup -----------------
arduino = None
arduino_connected = False
arduino_queue = queue.Queue(maxsize=30)

def connect_arduino():
    """Attempt connection to Arduino; ignore virtual Bluetooth ports to avoid hangs."""
    global arduino, arduino_connected
    try:
        import serial.tools.list_ports
        available_ports = list(serial.tools.list_ports.comports())
        target_port = None
        
        # Look for explicit USB-Serial / Arduino hardware
        for p in available_ports:
            desc = (p.description or "").lower()
            hwid = (p.hwid or "").lower()
            if "bluetooth" in desc or "bthenum" in hwid:
                continue
            if any(k in desc for k in ["arduino", "ch340", "cp210", "ftdi", "usb serial"]):
                target_port = p.device
                break
        
        if not target_port:
            for p in available_ports:
                if p.device.upper() == "COM3":
                    desc = (p.description or "").lower()
                    hwid = (p.hwid or "").lower()
                    if "bluetooth" not in desc and "bthenum" not in hwid:
                        target_port = "COM3"
                    break

        if target_port:
            import serial
            arduino = serial.Serial(target_port, 9600, timeout=0.5, write_timeout=0.5)
            time.sleep(1.5)
            print(f"✅ Arduino connected on {target_port}")
            arduino_connected = True
        else:
            print("ℹ️ No physical Arduino detected. Running in SIMULATION MODE.")
            arduino_connected = False
    except Exception as e:
        print(f"❌ Arduino connection skipped: {e}. Running in SIMULATION MODE.")
        arduino_connected = False

connect_arduino()

def arduino_worker():
    """Background worker thread to handle serial writes without freezing video playback."""
    while True:
        item = arduino_queue.get()
        if item is None:
            break
        command, animal, distance = item
        if arduino_connected and arduino is not None:
            try:
                data = {
                    'command': command,
                    'animal': animal,
                    'distance': round(distance, 1),
                    'unit': 'm',
                    'timestamp': int(time.time())
                }
                msg = json.dumps(data) + '\n'
                arduino.write(msg.encode())
                print(f"📤 Sent to Arduino: {command} ({animal} at {distance:.1f}m)")
                if arduino.in_waiting:
                    resp = arduino.readline().decode(errors='ignore').strip()
                    if resp:
                        print(f"📥 Arduino: {resp}")
            except Exception as e:
                print(f"❌ Serial communication error: {e}")
        else:
            print(f"[SIMULATION] {command} for {animal} at {distance:.1f}m")
        arduino_queue.task_done()

threading.Thread(target=arduino_worker, daemon=True).start()

def send_to_arduino(command, animal, distance):
    try:
        arduino_queue.put_nowait((command, animal, distance))
    except queue.Full:
        pass

# ----------------- Deterrent Cooldowns -----------------
last_sound_time = 0
last_pump_time = 0
sound_cooldown = 3.0   # seconds
pump_cooldown = 15.0   # seconds

def play_sound(animal):
    global last_sound_time
    now = time.time()
    if now - last_sound_time < sound_cooldown:
        return
    last_sound_time = now

    print(f"🔊 Playing {animal} deterrent sound")
    if animal == "cow" and cow_sound:
        cow_sound.play()
    elif animal == "elephant" and elephant_sound:
        elephant_sound.play()

def activate_pump(animal, distance):
    global last_pump_time
    now = time.time()
    if now - last_pump_time < pump_cooldown:
        remaining = pump_cooldown - (now - last_pump_time)
        print(f"💧 Pump cooldown active ({remaining:.1f}s remaining)")
        return
    
    last_pump_time = now
    send_to_arduino("PUMP_ON", animal, distance)
    print(f"💧 PUMP ACTIVATED! Deterring {animal} at {distance:.1f}m")

# ----------------- Distance Calculation (in Meters) -----------------
def calculate_distance(bbox_height, animal_name):
    """Calculate distance in meters using focal length method."""
    if bbox_height <= 0:
        return 99.9
    
    focal_length = 800
    real_heights = {
        "cow": 1.5,       # 1.5 meters (~150 cm)
        "elephant": 3.0,  # 3.0 meters (~300 cm)
        "person": 1.7,    # 1.7 meters (~170 cm)
    }
    real_height = real_heights.get(animal_name, 1.5)
    distance = (real_height * focal_length) / bbox_height
    return min(distance, 100.0)

def draw_detections(frame, detections, current_state, fps_display=30.0):
    """Draw bounding boxes, labels in meters, and a real-time HUD."""
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        distance = d['distance']
        confidence = d['confidence']
        label = f"{d['name'].upper()} {distance:.1f}m ({confidence:.0%})"

        # Color coding in meters:
        # < 10m: Red, < 30m: Orange, >= 30m: Green
        if distance < 10.0:
            color = (0, 0, 255)
            thickness = 3
        elif distance < 30.0:
            color = (0, 165, 255)
            thickness = 2
        else:
            color = (0, 255, 0)
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        cv2.rectangle(frame, (x1, max(0, y1 - label_size[1] - 8)), 
                             (x1 + label_size[0] + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    
    # Top HUD Banner
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    if current_state == "RED":
        status_text = f"ALERT: {len(detections)} DETECTED | {fps_display:.0f} FPS | 'Q' to quit"
        status_color = (0, 80, 255)
    else:
        status_text = f"CLEAR: SAFE | {fps_display:.0f} FPS | 'Q' to quit"
        status_color = (0, 220, 0)

    cv2.putText(frame, status_text, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2)
    
    return frame

# ----------------- Real-Time Async Video Player & Detector -----------------
# Shared thread state
latest_frame_for_ai = None
latest_detections = []
frame_lock = threading.Lock()
is_running = True

def ai_inference_worker():
    """Asynchronous AI thread: Analyzes latest frames without stalling 30 FPS video playback."""
    global latest_detections, is_running
    while is_running:
        frame_to_process = None
        with frame_lock:
            if latest_frame_for_ai is not None:
                frame_to_process = latest_frame_for_ai.copy()
        
        if frame_to_process is None:
            time.sleep(0.01)
            continue
            
        # Run inference (imgsz=480 provides fast CPU inference while keeping accuracy)
        results = model.predict(frame_to_process, conf=0.38, imgsz=480, verbose=False)
        new_detections = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    name = r.names[cls]
                    if name not in ["cow", "elephant"]:
                        continue
                        
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox_height = y2 - y1
                    distance = calculate_distance(bbox_height, name)
                    
                    new_detections.append({
                        "name": name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "distance": distance
                    })
                    
                    # Trigger deterrents
                    threading.Thread(target=play_sound, args=(name,), daemon=True).start()
                    if distance < 30.0:
                        threading.Thread(target=activate_pump, args=(name, distance), daemon=True).start()

        with frame_lock:
            latest_detections = new_detections

def detect_animals():
    global latest_frame_for_ai, is_running
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Could not open video file: {VIDEO_PATH}")
        return

    window_name = "🐄 Animal Detection & Deterrent System (Real-Time 30 FPS)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 480, 854)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval_sec = 1.0 / fps

    print("=" * 60)
    print("🚀 Animal Detection System Started (Smooth 30 FPS Mode)")
    print(f"📹 Source Video: {VIDEO_PATH}")
    print("🎯 Target Animals: cows, elephants")
    print("📏 Distance Unit: METERS (m)")
    print("⚡ Real-Time Mode: Video plays at native 30 FPS smoothly")
    print("⚠️  Press 'Q' inside the video window to quit")
    print("=" * 60)

    # Launch AI inference worker thread
    ai_thread = threading.Thread(target=ai_inference_worker, daemon=True)
    ai_thread.start()

    last_detection_time = 0.0
    current_alert_state = "GREEN"
    fps_measure_time = time.time()
    frames_rendered = 0
    current_fps = fps

    while is_running:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            # Loop video file continuously
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                print("❌ Unable to read video frames")
                break

        # Pass current frame to background AI thread
        with frame_lock:
            latest_frame_for_ai = frame
            active_detections = list(latest_detections)

        # Alert state management
        current_time = time.time()
        if len(active_detections) > 0:
            last_detection_time = current_time
            closest = min(active_detections, key=lambda d: d['distance'])
            if current_alert_state != "RED":
                print(f"🎯 {closest['name'].upper()} detected at {closest['distance']:.1f}m (confidence: {closest['confidence']:.0%})")
                send_to_arduino("RED", closest['name'], closest['distance'])
                current_alert_state = "RED"
        else:
            if current_alert_state == "RED" and (current_time - last_detection_time > 2.0):
                send_to_arduino("GREEN", "none", 99.9)
                current_alert_state = "GREEN"

        # Calculate live rendering FPS
        frames_rendered += 1
        if time.time() - fps_measure_time >= 1.0:
            current_fps = frames_rendered / (time.time() - fps_measure_time)
            frames_rendered = 0
            fps_measure_time = time.time()

        # Render detections & HUD at true video speed
        display_frame = draw_detections(frame, active_detections, current_alert_state, current_fps)
        cv2.imshow(window_name, display_frame)
        
        # Enforce exact 30 FPS timing
        elapsed_sec = time.time() - frame_start
        wait_ms = max(1, int((frame_interval_sec - elapsed_sec) * 1000.0))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            print("\n🛑 Quit key detected. Shutting down...")
            break

    is_running = False
    cap.release()
    cv2.destroyAllWindows()
    
    if arduino_connected and arduino:
        send_to_arduino("GREEN", "shutdown", 0.0)
        time.sleep(0.3)
        try:
            arduino.close()
        except Exception:
            pass
        print("🔌 Arduino connection closed")
    
    print("✅ System shutdown complete")

if __name__ == '__main__':
    try:
        detect_animals()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        is_running = False
        cv2.destroyAllWindows()
        if arduino_connected and arduino:
            try:
                arduino.close()
            except Exception:
                pass