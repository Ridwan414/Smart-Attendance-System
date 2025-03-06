import cv2
import os
import time
import numpy as np
import face_recognition
import joblib
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, jsonify
from datetime import datetime
from mtcnn import MTCNN
import csv
import serial
import serial.tools.list_ports
import re


app = Flask(__name__)

# Paths
KNOWN_FACES_DIR = "static/faces"
DATA_DIR = "data"

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Generate attendance file path with today's date
ATTENDANCE_CSV = f"{DATA_DIR}/Attendance-{datetime.today().strftime('%m_%d_%y')}.csv"
RFID_USER_CSV = os.path.join(DATA_DIR, "users.csv")

# Initialize RFID CSV files if they don't exist
if not os.path.exists(RFID_USER_CSV):
    with open(RFID_USER_CSV, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["RFID ID","User ID","Name"])
if not os.path.exists(ATTENDANCE_CSV):
    with open(ATTENDANCE_CSV, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["User ID","Name","Time"])

# Load saved encodings
if os.path.exists("face_encodings.pkl"):
    known_face_encodings, known_face_names = joblib.load("face_encodings.pkl")
else:
    known_face_encodings, known_face_names = [], []
rfid_users = {}

# Initialize MTCNN
detector = MTCNN()
# Face recognition threshold
FACE_MATCH_THRESHOLD = 0.45  # Stricter threshold for higher accuracy

# Add this after other global variables
RFID_PORT = None  # Will store the RFID reader's serial port

def init_rfid_reader():
    """Initialize RFID reader connection"""
    try:
        # Look for USB devices that might be the RFID reader
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            if ("AuthenTec" in port.description or 
                "SYC ID&IC USB Reader" in port.description or 
                "08FF:0009" in port.hwid.upper()):
                return True
                
        # Alternative method: check if hidraw1 exists in /dev
        if os.path.exists("/dev/hidraw1"):
            return True
            
        return False
    except Exception as e:
        print(f"Error checking RFID reader: {e}")
        return False

def read_rfid_card(timeout=15):
    """Read RFID card and return the ID"""
    print(f"⌛ Waiting for RFID card (timeout: {timeout}s)...")
    if not check_rfid_reader():
        print("❌ No RFID reader detected")
        return None
    
    # Open the HID device
    try:
        print("🔌 Opening RFID reader device...")
        rfid_port = open("/dev/hidraw1", "rb")
        print("✅ RFID reader opened successfully")
    except Exception as e:
        print(f"❌ Could not open RFID reader: {e}")
        return None
    
    # Read loop
    buffer = ""
    start_time = time.time()
    
    try:
        while (time.time() - start_time) < timeout:
            # Read data from HID device
            byte_data = rfid_port.read(8)
            
            if byte_data and any(b != 0 for b in byte_data):
                print(f"📡 Raw data received: {byte_data}")
                
                # Extract the third byte (index 2) which contains the actual digit
                if len(byte_data) > 2:
                    digit_byte = byte_data[2]
                    if 0x1E <= digit_byte <= 0x27:  # Map scan codes to digits
                        # Convert scan code to actual digit (0x1E = '1', 0x1F = '2', etc.)
                        digit = str(digit_byte - 0x1E + 1)
                        if digit_byte == 0x27:  # Special case for '0'
                            digit = '0'
                        buffer += digit
                        print(f"🔢 Current buffer: {buffer}")
                
                # Look for card ID pattern (sequence of digits)
                if len(buffer) >= 10:  # Wait for complete card number
                    print(f"💳 Found card ID: {buffer}")
                    rfid_port.close()
                    return buffer
            
            time.sleep(0.01)
        
        # Timeout reached
        print("⏰ Timeout waiting for card")
        return None
            
    except Exception as e:
        print(f"❌ Error reading RFID card: {e}")
        return None
    finally:
        print("🔌 Closing RFID reader")
        rfid_port.close()

# Function to load known faces
def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    # If no faces exist, clear encodings
    if not os.listdir(KNOWN_FACES_DIR):
        joblib.dump(([], []), "face_encodings.pkl")
        print("⚠️ No registered faces found. Resetting encodings.")
        return

    # Load new faces if available
    for person in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                img_path = os.path.join(person_dir, filename)
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(person)

    joblib.dump((known_face_encodings, known_face_names), "face_encodings.pkl")
    print(f"✅ Loaded {len(known_face_encodings)} registered faces.")



# Load faces if not already loaded
if not known_face_encodings:
    load_known_faces()


# Function to detect and recognize faces using MTCNN
def recognize_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_frame)

    recognized_names = []
    for detection in detections:
        x, y, width, height = detection['box']
        face_location = (y, x + width, y + height, x)

        face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])
        if not face_encoding:
            continue

        encoding = face_encoding[0]
        matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=FACE_MATCH_THRESHOLD)
        name = "Unknown"

        if True in matches:
            match_index = np.argmin(face_recognition.face_distance(known_face_encodings, encoding))
            name = known_face_names[match_index]

        recognized_names.append((name, (x, y, x + width, y + height)))

    return recognized_names


# Function to mark attendance
def mark_attendance(name):
    if name == "Unknown":
        return

    try:
        # Ensure the Attendance folder exists
        os.makedirs("Attendance", exist_ok=True)

        # Generate attendance file path with today's date
        attendance_file = ATTENDANCE_CSV
        # Check if file exists and load existing data
        if os.path.exists(attendance_file):
            df = pd.read_csv(attendance_file)
        else:
            df = pd.DataFrame(columns=["User ID", "Name", "Time"])

        current_time = datetime.now().strftime("%H:%M:%S")

        # Extract user ID from stored name format
        user_id = name.split("_")[-1]

        # Check if already marked
        if user_id not in df["User ID"].values:
            new_entry = pd.DataFrame({"User ID": [user_id], "Name": [name], "Time": [current_time]})
            df = pd.concat([df, new_entry], ignore_index=True)

            # Save and flush data immediately
            df.to_csv(attendance_file, index=False)
            print(f"✅ Attendance marked for {name} (User ID: {user_id}) at {current_time}")

        else:
            print(f"⚠️ {name} (User ID: {user_id}) is already marked present.")

    except Exception as e:
        print(f"❌ Error marking attendance: {e}")


def load_rfid_users():
    global rfid_users
    rfid_users = {}
    with open(RFID_USER_CSV, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 3:  # Make sure we have all columns
                rfid_id, user_id, name = row[0], row[1], row[2]
                rfid_users[rfid_id] = {"user_id": user_id, "name": name}  # Store both user_id and name

# Load RFID users at startup
load_rfid_users()

# Flask Routes

@app.route("/")
def home():
    # Count total registered users
    total_users = len(os.listdir(KNOWN_FACES_DIR)) if os.path.exists(KNOWN_FACES_DIR) else 0

    # Load latest attendance data
    attendance_data = pd.read_csv(ATTENDANCE_CSV) if os.path.exists(ATTENDANCE_CSV) else pd.DataFrame()

    return render_template(
        "home.html",
        total_users=total_users,  # Send total user count
        attendance=attendance_data.to_dict(orient="records")  # Send attendance data
    )







@app.route("/add", methods=["POST"])
def register_user():
    if "username" not in request.form or "user_id" not in request.form:
        return jsonify({"success": False, "message": "Error: Username and User ID required!"})

    username = request.form["username"].strip()
    user_id = request.form["user_id"].strip()

    if not username or not user_id:
        return jsonify({"success": False, "message": "Username and User ID cannot be empty!"})

    user_dir = os.path.join(KNOWN_FACES_DIR, f"{username}_{user_id}")
    os.makedirs(user_dir, exist_ok=True)

    try:
        cap = cv2.VideoCapture("/dev/video0")
        if not cap.isOpened():
            return jsonify({"success": False, "message": "Error: Could not access webcam!"})

        print(f"📸 Please position your face within the frame for {username} (User ID: {user_id})...")

        face_detected_frames = 0  # Counter for correctly positioned frames
        required_frames = 3  # User must hold still for 10 frames

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(rgb_frame)

            h, w, _ = frame.shape
            guide_x1, guide_y1, guide_x2, guide_y2 = int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)

            # Draw a guiding box for face alignment
            cv2.rectangle(frame, (guide_x1, guide_y1), (guide_x2, guide_y2), (255, 255, 0), 2)
            cv2.putText(frame, "Align your face within the blue box", (50, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            face_properly_aligned = False

            for detection in detections:
                x, y, width, height = detection['box']
                x2, y2 = x + width, y + height

                # Draw a bounding box around the detected face
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

                # Face is properly positioned if within the guiding box
                if guide_x1 < x and guide_x2 > x2 and guide_y1 < y and guide_y2 > y2 and 120 < width < 300:
                    face_properly_aligned = True
                    face_detected_frames += 1
                else:
                    face_detected_frames = 0  # Reset counter if face moves out of position

                # Display progress bar
                progress_width = int((face_detected_frames / required_frames) * 300)
                cv2.rectangle(frame, (50, h - 50), (50 + progress_width, h - 30), (0, 255, 0), -1)
                cv2.rectangle(frame, (50, h - 50), (350, h - 30), (255, 255, 255), 2)
                cv2.putText(frame, f"Hold still: {face_detected_frames}/{required_frames}", (50, h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Registering User - Adjust Your Position", frame)

            # If face is properly aligned for the required frames, capture images
            if face_detected_frames >= required_frames:
                print("✅ Face positioned correctly. Capturing images...")
                break

            # Press ESC to cancel registration
            if cv2.waitKey(1) & 0xFF == 27:
                print("🚪 User cancelled registration.")
                cap.release()
                cv2.destroyAllWindows()
                return jsonify({"success": False, "message": "Registration cancelled."})

        # Automatically capture 5 images with a delay
        for i in range(5):
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Failed to capture frame {i + 1}")
                continue

            # Display capture count
            cv2.putText(frame, f"Capturing Image {i + 1}/5", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Registering User", frame)
            cv2.imwrite(os.path.join(user_dir, f"{username}_{user_id}_{i}.jpg"), frame)

            time.sleep(3)  # Delay between captures
            if cv2.waitKey(1) & 0xFF == 27:
                print("🚪 User cancelled registration.")
                break

        # Load newly added faces
        load_known_faces()

        return jsonify({
            "success": True,
            "message": f"✅ {username} (User ID: {user_id}) successfully registered with 5 images!"
        })
    except Exception as e:
        print(f"❌ Camera error during registration: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})
    finally:
        # Ensure camera is properly released and windows are closed
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()





@app.route("/attendance/faces")
def start_attendance():
    global known_face_encodings, known_face_names

    # Reload faces in case they were cleared
    if not os.listdir(KNOWN_FACES_DIR):
        known_face_encodings, known_face_names = [], []
        print("⚠️ No registered faces. Only detecting unknown users.")

    try:
        cap = cv2.VideoCapture("/dev/video0")
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return redirect(url_for("home"))
            
        observed_faces = {}  # Track consistent recognition
        marked_attendance = set()  # Track users who already had attendance marked

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            recognized_faces = recognize_faces(frame)
            for name, (x1, y1, x2, y2) in recognized_faces:
                if not known_face_encodings:
                    name = "Unknown"  # Force unknown if no registered faces

                if name != "Unknown":
                    if name in observed_faces:
                        observed_faces[name]["count"] += 1
                        observed_faces[name]["last_seen"] = time.time()
                    else:
                        observed_faces[name] = {"count": 1, "last_seen": time.time()}

                    # Ensure face is recognized for at least 10 frames before marking attendance
                    if observed_faces[name]["count"] >= 10 and name not in marked_attendance:
                        mark_attendance(name)
                        marked_attendance.add(name)  # Add to set of marked users
                        print(f"✅ {name} confirmed and attendance marked!")

                # Set box color (Red for Unknown, Green for Recognized)
                box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                # Draw rectangle around the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

            cv2.imshow("Face Attendance System", frame)

            # Remove stale records after 15 seconds
            current_time = time.time()
            observed_faces = {k: v for k, v in observed_faces.items() if current_time - v["last_seen"] < 15} 

            # Check for ESC key press (ASCII 27)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # 27 is the ASCII code for ESC
                print("🚪 ESC key pressed. Exiting webcam & saving attendance...")
                break

    except Exception as e:
        print(f"❌ Camera error: {e}")
    finally:
        # Ensure camera is properly released and windows are closed
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    # Return to home page after attendance session ends
    return redirect(url_for("home"))


@app.route("/rfid/register", methods=["POST"])
def register_rfid():
    print("📝 Starting RFID registration...")
    user_id = request.form.get("user_id", "").strip()
    rfid_id = request.form.get("rfid_id", "").strip()
    name = request.form.get("name", "").strip()

    print(f"Received data: user_id={user_id}, rfid_id={rfid_id}, name={name}")

    if not user_id or not name:
        print("❌ Missing required fields")
        return jsonify({"error": "User ID and Name are required!"})

    if not rfid_id:
        print("❌ Missing RFID ID")
        return jsonify({"error": "RFID ID is required!"})

    try:
        # Check if RFID ID or User ID already exists
        with open(RFID_USER_CSV, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 3:  # Ensure row has all required fields
                    if row[0] == rfid_id:
                        print(f"❌ RFID ID {rfid_id} already exists")
                        return jsonify({"error": "This RFID card is already registered!"})
                    if row[1] == user_id:
                        print(f"❌ User ID {user_id} already exists")
                        return jsonify({"error": "This User ID is already registered!"})

        # Register new user
        with open(RFID_USER_CSV, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([rfid_id, user_id, name])
            print(f"✅ Successfully registered: {name} (ID: {user_id})")

        # Update local dictionary
        rfid_users[rfid_id] = {"user_id": user_id, "name": name}
        print("✅ Local dictionary updated")

        return jsonify({
            "success": True,
            "message": f"Registration successful for {name}!"
        })
    except Exception as e:
        print(f"❌ Error during registration: {e}")
        return jsonify({"error": f"Registration failed: {str(e)}"})

@app.route("/rfid/scan", methods=["POST"])
def rfid_scan():
    rfid_id = request.form.get("rfid_id", "").strip()
    
    if not rfid_id:
        return jsonify({"error": "Invalid scan! RFID ID is empty."})

    if rfid_id not in rfid_users:
        return jsonify({"error": "Unknown RFID card!"})

    user_data = rfid_users[rfid_id]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(ATTENDANCE_CSV, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([user_data["user_id"], user_data["name"], timestamp])

    return jsonify({"message": f"Attendance recorded for {user_data['name']}!"})

@app.route("/check_rfid", methods=["GET"])
def check_rfid_available():
    """Endpoint to check RFID reader availability"""
    return jsonify({
        "available": check_rfid_reader(),
        "message": "RFID reader detected" if check_rfid_reader() else "No RFID reader found"
    })

def check_rfid_reader():
    """Check if RFID reader is connected via USB"""
    try:
        return init_rfid_reader()
    except:
        return False

@app.route("/rfid/register_only", methods=["POST"])
def register_rfid_only():
    print("📝 Starting RFID-only registration...")
    user_id = request.form.get("user_id", "").strip()
    rfid_id = request.form.get("rfid_id", "").strip()
    name = request.form.get("name", "").strip()

    print(f"Received data: user_id={user_id}, rfid_id={rfid_id}, name={name}")

    if not user_id or not name or not rfid_id:
        return jsonify({"error": "User ID, Name, and RFID ID are required!"})

    try:
        # Check if RFID ID or User ID already exists
        with open(RFID_USER_CSV, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 3:  # Ensure row has all required fields
                    if row[0] == rfid_id:
                        print(f"❌ RFID ID {rfid_id} already exists")
                        return jsonify({"error": "This RFID card is already registered!"})
                    if row[1] == user_id:
                        print(f"❌ User ID {user_id} already exists")
                        return jsonify({"error": "This User ID is already registered!"})

        # Register new user
        with open(RFID_USER_CSV, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([rfid_id, user_id, name])
            print(f"✅ Successfully registered: {name} (ID: {user_id})")

        # Update local dictionary
        rfid_users[rfid_id] = {"user_id": user_id, "name": name}
        print("✅ Local dictionary updated")

        return jsonify({
            "success": True,
            "message": f"Registration successful for {name}!"
        })
    except Exception as e:
        print(f"❌ Error during registration: {e}")
        return jsonify({"error": f"Registration failed: {str(e)}"})

@app.route("/rfid/read", methods=["GET"])
def read_rfid():
    """Endpoint to read RFID card"""
    print("🔍 Starting RFID card read...")
    try:
        card_id = read_rfid_card()
        if card_id:
            print(f"✅ Card read successfully: {card_id}")
            return jsonify({
                "success": True,
                "rfid_id": card_id
            })
        print("❌ No card detected within timeout period")
        return jsonify({
            "success": False,
            "error": "No card detected within timeout period"
        })
    except Exception as e:
        print(f"❌ Error reading RFID card: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route("/attendance/rfid")
def start_rfid_attendance():
    """Start RFID attendance monitoring"""
    try:
        print("🔄 Starting RFID attendance mode...")
        while True:
            card_id = read_rfid_card(timeout=1)  # Short timeout for continuous reading
            if card_id:
                print(f"💳 Card detected: {card_id}")
                if card_id in rfid_users:
                    user_data = rfid_users[card_id]
                    
                    # Check if already marked attendance today
                    with open(ATTENDANCE_CSV, mode="r") as file:
                        reader = csv.reader(file)
                        next(reader)  # Skip header
                        already_marked = False
                        for row in reader:
                            if row[0] == user_data["user_id"]:  # Check User ID
                                already_marked = True
                                print(f"⚠️ {user_data['name']} already marked attendance today")
                                break
                    
                    if not already_marked:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        
                        with open(ATTENDANCE_CSV, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([user_data["user_id"], user_data["name"], timestamp])
                        
                        print(f"✅ Attendance marked for {user_data['name']}")
                else:
                    print("❌ Unknown RFID card")
                
                time.sleep(1)  # Prevent multiple reads of the same card
            
            # Check for ESC key press
            if cv2.waitKey(1) & 0xFF == 27:  # 27 is ESC key
                print("\n🛑 ESC pressed - stopping RFID attendance")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 RFID attendance monitoring stopped")
    except Exception as e:
        print(f"❌ Error in RFID attendance: {e}")
    finally:
        cv2.destroyAllWindows()
    
    return redirect(url_for("home"))

@app.route("/get_attendance")
def get_attendance():
    """Endpoint to get current attendance data"""
    try:
        if os.path.exists(ATTENDANCE_CSV):
            df = pd.read_csv(ATTENDANCE_CSV)
            return jsonify({
                "success": True,
                "attendance": df.to_dict(orient="records")
            })
        return jsonify({
            "success": True,
            "attendance": []
        })
    except Exception as e:
        print(f"❌ Error getting attendance data: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get attendance data"
        })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
