import cv2
import os
import time
import numpy as np
import face_recognition
import joblib
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from datetime import datetime
from mtcnn import MTCNN

# Initialize Flask app
app = Flask(__name__)

# Paths
KNOWN_FACES_DIR = "static/faces"
ATTENDANCE_CSV = f"Attendance/Attendance-{datetime.today().strftime('%m_%d_%y')}.csv"

# Create directories if they don’t exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs("Attendance", exist_ok=True)

# Initialize MTCNN
detector = MTCNN()

# Face recognition threshold
FACE_MATCH_THRESHOLD = 0.45  # Stricter threshold for higher accuracy

# Load saved encodings
if os.path.exists("face_encodings.pkl"):
    known_face_encodings, known_face_names = joblib.load("face_encodings.pkl")
else:
    known_face_encodings, known_face_names = [], []

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
    
    # Detect face locations using the CNN model for better accuracy
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    
    recognized_names = []
    for face_location in face_locations:
        top, right, bottom, left = face_location
        
        # Extract the face encoding
        face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])
        if not face_encoding:
            continue

        # Check if the face is frontal using landmarks
        landmarks = face_recognition.face_landmarks(rgb_frame, [face_location])
        if not landmarks:
            continue

        # Ensure that the face is frontal by checking the presence of key landmarks
        if 'chin' not in landmarks[0] or 'nose_bridge' not in landmarks[0]:
            continue

        encoding = face_encoding[0]
        matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=FACE_MATCH_THRESHOLD)
        name = "Unknown"

        if True in matches:
            match_index = np.argmin(face_recognition.face_distance(known_face_encodings, encoding))
            name = known_face_names[match_index]

        recognized_names.append((name, (left, top, right, bottom)))

    return recognized_names

# Function to mark attendance
def mark_attendance(name):
    print(f"Attempting to mark attendance for {name}...")
    if name == "Unknown":
        return

    try:
        # Ensure the Attendance folder exists
        os.makedirs("Attendance", exist_ok=True)

        # Generate attendance file path with today's date
        attendance_file = f"Attendance/Attendance-{datetime.today().strftime('%m_%d_%y')}.csv"

        # Check if file exists and load existing data
        if os.path.exists(attendance_file):
            df = pd.read_csv(attendance_file)
            print(f"Attendance file {attendance_file} exists.")
        else:
            df = pd.DataFrame(columns=["User  ID", "Name", "Time"])
            print(f"Attendance file {attendance_file} does not exist. Creating a new one.")

        current_time = datetime.now().strftime("%H:%M:%S")

        # Extract user ID from stored name format
        user_id = name.split("_")[-1]

        # Check if already marked
        if user_id not in df["User  ID"].values:
            new_entry = pd.DataFrame({"User  ID": [user_id], "Name": [name], "Time": [current_time]})
            df = pd.concat([df, new_entry], ignore_index=True)

            # Save and flush data immediately
            df.to_csv(attendance_file, index=False)
            print(f"✅ Attendance marked for {name} (User  ID: {user_id}) at {current_time}")

        else:
            print(f"⚠️ {name} (User  ID: {user_id}) is already marked present.")

    except Exception as e:
        print(f"❌ Error marking attendance: {e}")

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
        return render_template("home.html", message="Error: Username and User ID required!")

    username = request.form["username"].strip()
    user_id = request.form["user_id"].strip()

    if not username or not user_id:
        return render_template("home.html", message="Username and User ID cannot be empty!")

    user_dir = os.path.join(KNOWN_FACES_DIR, f"{username}_{user_id}")
    os.makedirs(user_dir, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend for better Windows performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set fixed resolution for consistency
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Maximize FPS for smooth video

    if not cap.isOpened():
        return render_template("home.html", message="Error: Could not access webcam!")

    print(f"📸 Capturing 100 smooth images for {username} (User ID: {user_id})...")

    count = 0
    frame_skip = 5  # Only detect face every 5 frames to avoid lag
    frame_count = 0

    while count < 100:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face every `frame_skip` frames to reduce CPU load
        if frame_count % frame_skip == 0:
            faces = detector.detect_faces(rgb_frame)

        for face in faces:
            x, y, width, height = face["box"]
            x, y = max(0, x), max(0, y)

            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Display capture count on the top-left
            cv2.putText(frame, f"Capturing {count}/100", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Ensure face is large enough before saving
            if width > 80 and height > 80:
                cropped_face = rgb_frame[y:y+height, x:x+width]
                resized_face = cv2.resize(cropped_face, (150, 150))  # Standard size for face recognition
                cv2.imwrite(os.path.join(user_dir, f"{username}_{user_id}_{count}.jpg"), 
                            cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR))
                count += 1
                print(f"✅ Captured {count}/100 images")

        frame_count += 1

        # Show real-time capture preview
        cv2.imshow("Ultra-Smooth Face Registration", frame)

        # Press ESC to exit early
        if cv2.waitKey(1) & 0xFF == 27:
            print("🚪 Registration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return render_template("home.html", message="Registration cancelled.")

    cap.release()
    cv2.destroyAllWindows()

    print(f"✅ {username} (User ID: {user_id}) registered with 100 smooth images!")

    # Load newly added faces
    load_known_faces()

    return render_template("home.html", message=f"✅ {username} registered with ultra-smooth capture!")

@app.route("/attendance/faces")
def start_attendance():
    global known_face_encodings, known_face_names

    # Reload faces in case they were cleared
    if not os.listdir(KNOWN_FACES_DIR):
        known_face_encodings, known_face_names = [], []
        print("⚠️ No registered faces. Only detecting unknown users.")

    cap = cv2.VideoCapture(0)
    observed_faces = {}  # Track consistent recognition

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        recognized_faces = recognize_faces(frame)
        print("Recognized faces:", recognized_faces)  # Debugging output

        for name, (x1, y1, x2, y2) in recognized_faces:
            print(f"Processing face: {name}")  # Debugging output
            if not known_face_encodings:
                name = "Unknown"  # Force unknown if no registered faces

            if name != "Unknown":
                if name in observed_faces:
                    observed_faces[name]["count"] += 1
                    observed_faces[name]["last_seen"] = time.time()
                else:
                    observed_faces[name] = {"count": 1, "last_seen": time.time()}

                # Ensure face is recognized for at least 10 seconds before marking attendance
                if observed_faces[name]["count"] >= 10:  # Adjust this based on your frame rate
                    mark_attendance(name)
                    print(f"✅ {name} confirmed and attendance marked!")

            # Set box color (Red for Unknown, Green for Recognized)
            box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        cv2.imshow("Face Attendance System", frame)

        # Remove stale records after 10 seconds
        current_time = time.time()
        observed_faces = {k: v for k, v in observed_faces.items() if current_time - v["last_seen"] < 15}

        # Check for ESC key press (ASCII 27)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 27 is the ASCII code for ESC
            print("🚪 ESC key pressed. Exiting webcam & saving attendance...")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Force page to refresh after attendance is marked
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)