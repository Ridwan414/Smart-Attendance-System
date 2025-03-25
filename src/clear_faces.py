import os
import shutil

def clear_faces():
    faces_dir = 'static/faces'
    if os.path.isdir(faces_dir):
        # Remove all folders and files under 'faces'
        shutil.rmtree(faces_dir)
        # Recreate the empty 'faces' directory
        os.makedirs(faces_dir)
        print("Faces directory cleared.")
    else:
        print("Faces directory does not exist.")

if __name__ == "__main__":
    clear_faces()
