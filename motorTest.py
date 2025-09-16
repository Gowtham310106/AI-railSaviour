# Animal Detection System - Updated for Pump Integration
try:
    import serial
    print(f"Serial module version: {serial.__version__}")
    print(f"Available attributes: {dir(serial)}")
    # Test creating a Serial object
    test_serial = serial.Serial()
    test_serial.close()
    print("✅ Serial module working correctly")
except Exception as e:
    print(f"❌ Serial module error: {e}")
import cv2
import time
import json
import threading
import pygame
#import serial
from ultralytics import YOLO

# Initialize Pygame mixer
pygame.mixer.init()

# Load deterrent animal sounds
cow_sound = pygame.mixer.Sound("animal_sounds/cow.mp3")
elephant_sound = pygame.mixer.Sound("animal_sounds/beebuzz.mp3")

# Load YOLO model
model = YOLO("yolov8m.pt")

# Connect to Arduino
try:
    arduino = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2)
    print("✅ Arduino connected")
    arduino_connected = True
except Exception as e:
    print(f"❌ Arduino not connected: {e}")
    arduino_connected = False

# Cooldowns
last_sound_time = 0
last_pump_time = 0
sound_cooldown = 3  # seconds
pump_cooldown = 15  # seconds (increased due to Arduino auto-shutoff)

# Video input
VIDEO_PATH = "videos/cowvid.mp4"


def play_sound(animal):
    global last_sound_time
    now = time.time()
    if now - last_sound_time < sound_cooldown:
        print(f"🔇 Sound cooldown active. Skipped.")
        return
    last_sound_time = now

    print(f"🔊 Playing {animal} deterrent sound")
    if animal == "cow":
        cow_sound.play()
    elif animal == "elephant":
        elephant_sound.play()

def send_to_arduino(command, animal, distance):
    if not arduino_connected:
        print(f"[SIMULATION] {command} for {animal} at {distance:.1f}cm")
        return
    try:
        data = {
            'command': command,
            'animal': animal,
            'distance': round(distance, 1),
            'timestamp': int(time.time())
        }
        message = json.dumps(data) + '\n'
        arduino.write(message.encode())
        print(f"📤 Sent to Arduino: {command} ({animal} at {distance:.1f}cm)")
        
        # Read Arduino response with timeout
        start_time = time.time()
        while time.time() - start_time < 0.5:  # 500ms timeout
            if arduino.in_waiting:
                response = arduino.readline().decode().strip()
                if response:
                    print(f"📥 Arduino: {response}")
                    break
                    
    except Exception as e:
        print(f"❌ Serial communication error: {e}")

def activate_pump(animal, distance):
    global last_pump_time
    now = time.time()
    if now - last_pump_time < pump_cooldown:
        remaining = pump_cooldown - (now - last_pump_time)
        print(f"💧 Pump cooldown active ({remaining:.1f}s remaining)")
        return
    
    last_pump_time = now
    send_to_arduino("PUMP_ON", animal, distance)
    print(f"💧 PUMP ACTIVATED! Deterring {animal} at {distance:.1f}cm")

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on detected animals"""
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        distance = d['distance']
        confidence = d['confidence']
        label = f"{d['name'].upper()} {distance:.1f}cm ({confidence:.1%})"

        # Color coding based on distance
        if distance < 1000:
            color = (0, 0, 255)      # Red - Very close!
            thickness = 3
        elif distance < 3000:
            color = (0, 165, 255)    # Orange - Close
            thickness = 2
        else:
            color = (0, 255, 0)      # Green - Far
            thickness = 2

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label with background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add status text
    status_text = f"Detections: {len(detections)} | Press 'Q' to quit"
    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def calculate_distance(bbox_height, animal_name):
    """Calculate distance using focal length method"""
    if bbox_height <= 0:
        return 9999
    
    # Camera focal length (calibrate for your camera)
    focal_length = 800
    
    # Real world heights in cm
    real_heights = {
        "cow": 150,      # Average cow height
        "elephant": 300,  # Average elephant height
        "person": 170,    # Average person height (if you want to add)
    }
    
    real_height = real_heights.get(animal_name, 150)  # Default to cow height
    distance = (real_height * focal_length) / bbox_height
    
    # Cap maximum distance for realistic values
    return min(distance, 10000)

def detect_animals():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Could not open camera/video")
        return

    print("🚀 Animal Detection System Started")
    print("📹 Monitoring for cows and elephants...")
    print("🎯 Detection threshold: 50% confidence")
    print("💧 Pump activation distance: < 300cm")
    print("⚠️  Press 'Q' to quit")
    print("-" * 50)

    frame_count = 0
    last_detection_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # If using video file, loop it
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        
        # Run detection every frame (you can skip frames for performance)
        results = model.predict(frame, conf=0.5, verbose=False)
        detections = []
        current_time = time.time()

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    name = r.names[cls]
                    
                    # Only process target animals
                    if name not in ["cow", "elephant"]:
                        continue
                        
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox_height = y2 - y1
                    
                    # Calculate distance
                    distance = calculate_distance(bbox_height, name)
                    
                    detection = {
                        "name": name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "distance": distance
                    }
                    detections.append(detection)
                    
                    # Immediate actions for detected animals
                    print(f"🎯 {name.upper()} detected at {distance:.1f}cm (confidence: {confidence:.1%})")
                    
                    # Always turn on red LED when animal detected
                    send_to_arduino("RED", name, distance)
                    
                    # Play sound deterrent
                    threading.Thread(target=play_sound, args=(name,), daemon=True).start()
                    
                    # Activate pump if animal is close (< 300cm)
                    if distance < 3000:
                        print(f"⚠️  CLOSE ANIMAL! Activating deterrent systems!")
                        threading.Thread(target=activate_pump, args=(name, distance), daemon=True).start()
                    
                    last_detection_time = current_time

        # If no animals detected and it's been a while, send GREEN signal
        if len(detections) == 0:
            # Wait 2 seconds after last detection before sending GREEN
            if current_time - last_detection_time > 2:
                send_to_arduino("GREEN", "none", 9999)

        # Draw detections on frame
        frame = draw_detections(frame, detections)
        
        # Show frame
        cv2.imshow("🐄 Animal Detection & Deterrent System", frame)
        
        # Check for quit command
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n🛑 Shutting down system...")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    if arduino_connected:
        # Send final GREEN signal and close connection
        send_to_arduino("GREEN", "shutdown", 0)
        time.sleep(0.5)
        arduino.close()
        print("🔌 Arduino connection closed")
    
    print("✅ System shutdown complete")

if __name__ == '__main__':
    try:
        detect_animals()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        cv2.destroyAllWindows()
        if arduino_connected:
            arduino.close()