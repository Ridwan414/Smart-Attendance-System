import os

def clear_attendance():
    attendance_dir = os.path.join(os.path.dirname(__file__), 'data')
    if os.path.isdir(attendance_dir):
        # Iterate through and remove all files in 'data'
        for file in os.listdir(attendance_dir):
            file_path = os.path.join(attendance_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("Attendance data directory cleared.")
    else:
        print("Attendance data directory does not exist.")

if __name__ == "__main__":
    clear_attendance()
