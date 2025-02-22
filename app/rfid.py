from flask import Flask, render_template, request, jsonify
import csv
import os
from datetime import datetime

app = Flask(__name__)

# Directory to store CSV files
csv_directory = "attendance_records"
os.makedirs(csv_directory, exist_ok=True)

# CSV file paths
user_csv_file = os.path.join(csv_directory, "users.csv")
attendance_csv_file = os.path.join(csv_directory, "attendance.csv")

# Create CSV files if they don't exist
if not os.path.exists(user_csv_file):
    with open(user_csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["User ID", "Name"])

if not os.path.exists(attendance_csv_file):
    with open(attendance_csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["User ID", "Name", "Timestamp"])

# Load users into memory
users = {}
def load_users():
    global users
    users = {}
    with open(user_csv_file, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            users[row[0]] = row[1]

# Load users at startup
load_users()

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["POST"])
def register():
    user_id = request.form.get("user_id", "").strip()
    name = request.form.get("name", "").strip()

    if not user_id or not name:
        return jsonify({"error": "User ID and Name are required!"})

    if user_id in users:
        return jsonify({"error": "User ID already registered!"})

    with open(user_csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([user_id, name])

    users[user_id] = name  # Update local dictionary

    return jsonify({"message": "User registered successfully!"})

@app.route("/scan", methods=["POST"])
def scan():
    user_id = request.form.get("user_id", "").strip()
    print(user_id)
    
    if not user_id:
        return jsonify({"error": "Invalid scan! User ID is empty."})

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    name = users.get(user_id, "Unknown")

    with open(attendance_csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([user_id, name, timestamp])

    return jsonify({"message": f"Attendance recorded for {name}!"})

if __name__ == "__main__":
    app.run(debug=True)
