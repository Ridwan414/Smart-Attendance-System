import cv2
import os
import time
import numpy as np
import face_recognition
import joblib
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_file
from datetime import datetime
from mtcnn import MTCNN
import csv
import serial
import serial.tools.list_ports
import re
import io
import xlsxwriter


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

# Add this for session tracking
CURRENT_SESSION = None
SESSION_START_TIME = None

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


# Function to generate a new session file
def create_new_session():
    global CURRENT_SESSION, SESSION_START_TIME
    SESSION_START_TIME = datetime.now()
    session_id = SESSION_START_TIME.strftime("%m_%d_%y_%H_%M")
    CURRENT_SESSION = f"{DATA_DIR}/Session-{session_id}.csv"
    
    # Create the session file with headers
    if not os.path.exists(CURRENT_SESSION):
        with open(CURRENT_SESSION, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["User ID", "Name", "Time"])
    
    return CURRENT_SESSION

# Function to mark attendance
def mark_attendance(name):
    if name == "Unknown":
        return

    try:
        # Ensure we have an active session
        global CURRENT_SESSION
        if not CURRENT_SESSION:
            print("⚠️ No active session found. Creating a new one.")
            create_new_session()

        # Check if file exists and load existing data
        if os.path.exists(CURRENT_SESSION):
            df = pd.read_csv(CURRENT_SESSION)
        else:
            df = pd.DataFrame(columns=["User ID", "Name", "Time"])

        current_time = datetime.now().strftime("%H:%M:%S")

        # Extract user ID from stored name format
        user_id = name.split("_")[-1]

        # Check if already marked in this session
        if user_id not in df["User ID"].values:
            new_entry = pd.DataFrame({"User ID": [user_id], "Name": [name], "Time": [current_time]})
            df = pd.concat([df, new_entry], ignore_index=True)

            # Save and flush data immediately
            df.to_csv(CURRENT_SESSION, index=False)
            print(f"✅ Attendance marked for {name} (User ID: {user_id}) at {current_time}")

        else:
            print(f"⚠️ {name} (User ID: {user_id}) is already marked present in this session.")

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

    # Load latest attendance data from current session if available
    attendance_data = pd.DataFrame()
    if CURRENT_SESSION and os.path.exists(CURRENT_SESSION):
        attendance_data = pd.read_csv(CURRENT_SESSION)
    elif os.path.exists(ATTENDANCE_CSV):
        attendance_data = pd.read_csv(ATTENDANCE_CSV)

    # Get list of all session files
    session_files = []
    for file in os.listdir(DATA_DIR):
        if file.startswith("Session-"):
            try:
                # Extract datetime from filename
                session_time = file.replace("Session-", "").replace(".csv", "")
                dt = datetime.strptime(session_time, "%m_%d_%y_%H_%M")
                
                session_files.append({
                    "filename": file,
                    "display_name": dt.strftime("%b %d, %Y at %I:%M %p"),
                    "timestamp": dt.timestamp()  # Add timestamp for sorting
                })
            except Exception as e:
                print(f"Error parsing session file {file}: {e}")
                continue
    
    # Sort sessions by timestamp (newest first)
    session_files.sort(key=lambda x: x["timestamp"], reverse=True)

    return render_template(
        "home.html",
        total_users=total_users,
        attendance=attendance_data.to_dict(orient="records"),
        sessions=session_files,
        current_session=CURRENT_SESSION.split("/")[-1] if CURRENT_SESSION else None
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

    # Define the angles we want to capture
    angles = [
        "front facing",
        "slightly right",
        "slightly left",
        "slightly up",
        "slightly down"
    ]

    try:
        cap = cv2.VideoCapture("/dev/video3")
        if not cap.isOpened():
            return jsonify({"success": False, "message": "Error: Could not access webcam!"})

        print(f"📸 Starting face registration for {username} (User ID: {user_id})...")

        face_detected_frames = 0
        required_frames = 3
        capture_phase = False
        current_angle_index = 0
        last_capture_time = 0
        capture_delay = 3

        while current_angle_index < len(angles):
            ret, frame = cap.read()
            if not ret:
                continue

            display_frame = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(rgb_frame)

            h, w, _ = frame.shape
            guide_x1, guide_y1, guide_x2, guide_y2 = int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)

            # Draw guide box
            cv2.rectangle(display_frame, (guide_x1, guide_y1), (guide_x2, guide_y2), (255, 255, 0), 2)

            # Display current angle instruction
            current_angle = angles[current_angle_index]
            if not capture_phase:
                instruction = f"Position {current_angle} and stay still"
                cv2.putText(display_frame, instruction, 
                          (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(display_frame, f"Capturing {current_angle} image...", 
                          (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            face_properly_aligned = False

            for detection in detections:
                x, y, width, height = detection['box']
                x2, y2 = x + width, y + height

                # Draw face detection box
                box_color = (0, 255, 0) if guide_x1 < x and guide_x2 > x2 and guide_y1 < y and guide_y2 > y2 else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x2, y2), box_color, 2)

                if guide_x1 < x and guide_x2 > x2 and guide_y1 < y and guide_y2 > y2 and 120 < width < 300:
                    face_properly_aligned = True
                    if not capture_phase:
                        face_detected_frames += 1
                else:
                    face_detected_frames = max(0, face_detected_frames - 1)

            # Draw progress bar for alignment phase
            if not capture_phase:
                progress_width = int((face_detected_frames / required_frames) * 300)
                cv2.rectangle(display_frame, (50, h - 50), (50 + progress_width, h - 30), (0, 255, 0), -1)
                cv2.rectangle(display_frame, (50, h - 50), (350, h - 30), (255, 255, 255), 2)
                cv2.putText(display_frame, f"Hold still: {face_detected_frames}/{required_frames}", 
                          (50, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show overall progress
            cv2.putText(display_frame, f"Angle {current_angle_index + 1} of {len(angles)}", 
                      (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Start capture phase if face aligned long enough
            if face_detected_frames >= required_frames and not capture_phase:
                capture_phase = True
                last_capture_time = time.time()
                print(f"✅ Face positioned correctly for {current_angle}. Capturing...")

            # Handle image capture
            if capture_phase and time.time() - last_capture_time > capture_delay:
                if face_properly_aligned:
                    # Save image with angle information
                    filename = f"{username}_{user_id}_{current_angle.replace(' ', '_')}.jpg"
                    cv2.imwrite(os.path.join(user_dir, filename), frame)
                    print(f"📸 Captured {current_angle} image")
                    
                    # Reset for next angle
                    current_angle_index += 1
                    capture_phase = False
                    face_detected_frames = 0
                    
                    if current_angle_index < len(angles):
                        print(f"👉 Please position your face {angles[current_angle_index]}")
                        time.sleep(1)  # Brief pause between angles

            cv2.imshow("Registration - Follow Instructions", display_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                print("🚪 User cancelled registration.")
                cap.release()
                cv2.destroyAllWindows()
                return jsonify({"success": False, "message": "Registration cancelled."})

        print("✅ All angles captured successfully!")
        time.sleep(1)  # Brief pause to show completion

        # Load newly added faces
        load_known_faces()

        return jsonify({
            "success": True,
            "message": f"✅ {username} (User ID: {user_id}) successfully registered with {len(angles)} angles!"
        })

    except Exception as e:
        print(f"❌ Camera error during registration: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()





@app.route("/attendance/faces")
def start_attendance():
    global known_face_encodings, known_face_names, CURRENT_SESSION

    # Create a new session when starting attendance
    CURRENT_SESSION = create_new_session()
    print(f"📝 Starting new attendance session: {CURRENT_SESSION}")

    # Reload faces in case they were cleared
    if not os.listdir(KNOWN_FACES_DIR):
        known_face_encodings, known_face_names = [], []
        print("⚠️ No registered faces. Only detecting unknown users.")

    try:
        cap = cv2.VideoCapture("/dev/video3")
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

            # Display session info on the frame
            session_info = f"Session: {CURRENT_SESSION.split('/')[-1]}"
            cv2.putText(frame, session_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display elapsed time
            elapsed = datetime.now() - SESSION_START_TIME
            elapsed_str = f"Time: {elapsed.seconds // 60}m {elapsed.seconds % 60}s"
            cv2.putText(frame, elapsed_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
    """Handle RFID card scan"""
    rfid_id = request.form.get("rfid_id", "").strip()
    
    if not rfid_id:
        return jsonify({"error": "Invalid scan! RFID ID is empty."})

    if rfid_id not in rfid_users:
        return jsonify({"error": "Unknown RFID card!"})

    user_data = rfid_users[rfid_id]
    
    try:
        # Ensure we have an active session
        global CURRENT_SESSION
        if not CURRENT_SESSION:
            print("⚠️ No active session found. Creating a new one.")
            create_new_session()

        # Check if already marked in this session
        if os.path.exists(CURRENT_SESSION):
            df = pd.read_csv(CURRENT_SESSION)
            if user_data["user_id"] in df["User ID"].values:
                return jsonify({"error": f"{user_data['name']} already marked in this session"})

        # Mark attendance
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(CURRENT_SESSION, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([user_data["user_id"], user_data["name"], timestamp])

        print(f"✅ Attendance marked for {user_data['name']}")
        return jsonify({
            "success": True,
            "message": f"Attendance recorded for {user_data['name']}!",
            "user": {
                "id": user_data["user_id"],
                "name": user_data["name"],
                "time": timestamp
            }
        })

    except Exception as e:
        print(f"❌ Error recording attendance: {e}")
        return jsonify({"error": f"Failed to record attendance: {str(e)}"})

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
    global CURRENT_SESSION
    
    try:
        # Create a new session
        CURRENT_SESSION = create_new_session()
        print(f"📝 Starting new RFID attendance session: {CURRENT_SESSION}")
        
        # Return success to indicate session started
        return jsonify({
            "success": True,
            "message": "RFID attendance session started",
            "session": CURRENT_SESSION.split("/")[-1]
        })
        
    except Exception as e:
        print(f"❌ Error starting RFID attendance session: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to start RFID attendance session"
        })

@app.route("/get_attendance")
def get_attendance():
    """Endpoint to get current attendance data"""
    try:
        session_id = request.args.get("session_id", None)
        
        if session_id:
            # Load specific session
            session_file = os.path.join(DATA_DIR, session_id)
            if os.path.exists(session_file):
                df = pd.read_csv(session_file)
                return jsonify({
                    "success": True,
                    "attendance": df.to_dict(orient="records"),
                    "session_name": session_id
                })
        elif CURRENT_SESSION and os.path.exists(CURRENT_SESSION):
            # Load current session
            df = pd.read_csv(CURRENT_SESSION)
            return jsonify({
                "success": True,
                "attendance": df.to_dict(orient="records"),
                "session_name": CURRENT_SESSION.split("/")[-1]
            })
        elif os.path.exists(ATTENDANCE_CSV):
            # Fall back to default attendance file
            df = pd.read_csv(ATTENDANCE_CSV)
            return jsonify({
                "success": True,
                "attendance": df.to_dict(orient="records"),
                "session_name": "Default"
            })
            
        return jsonify({
            "success": True,
            "attendance": [],
            "session_name": "No active session"
        })
    except Exception as e:
        print(f"❌ Error getting attendance data: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get attendance data"
        })

@app.route("/get_sessions")
def get_sessions():
    """Endpoint to get list of all sessions"""
    try:
        session_files = []
        for file in os.listdir(DATA_DIR):
            if file.startswith("Session-"):
                try:
                    # Extract datetime from filename
                    session_time = file.replace("Session-", "").replace(".csv", "")
                    dt = datetime.strptime(session_time, "%m_%d_%y_%H_%M")
                    
                    session_files.append({
                        "filename": file,
                        "display_name": dt.strftime("%b %d, %Y at %I:%M %p"),
                        "timestamp": dt.timestamp()  # Add timestamp for sorting
                    })
                except Exception as e:
                    print(f"Error parsing session file {file}: {e}")
                    continue
        
        # Sort sessions by timestamp (newest first)
        session_files.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return jsonify({
            "success": True,
            "sessions": session_files
        })
    except Exception as e:
        print(f"❌ Error getting sessions: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get sessions"
        })

@app.route("/download/<filename>")
def download_file(filename):
    """Download attendance file in Excel format"""
    try:
        # Determine the file path based on filename
        if filename.startswith("Session-"):
            file_path = os.path.join(DATA_DIR, filename)
        elif filename.startswith("Attendance-"):
            file_path = os.path.join(DATA_DIR, filename)
        else:
            return jsonify({"error": "Invalid file name"}), 400

        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        # Read CSV file
        df = pd.read_csv(file_path)

        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Attendance', index=False)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Attendance']
            
            # Add some formatting
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4B5563',
                'font_color': 'white',
                'border': 1
            })
            
            # Format headers
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            # Auto-adjust columns width
            for column in df:
                column_length = max(df[column].astype(str).apply(len).max(), len(column))
                col_idx = df.columns.get_loc(column)
                worksheet.set_column(col_idx, col_idx, column_length + 2)

        output.seek(0)
        
        # Generate Excel filename
        excel_filename = filename.replace('.csv', '.xlsx')
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=excel_filename
        )

    except Exception as e:
        print(f"Error downloading file: {e}")
        return jsonify({"error": "Failed to download file"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
