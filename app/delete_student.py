import os
import shutil

def delete_student_folder():
    # Ask for student name and ID
    student_name = input("Enter the student name: ")
    student_id = input("Enter the student ID: ")

    # Construct the folder name using the name and ID
    folder_name = f"static/faces/{student_name}_{student_id}"

    # Check if the folder exists
    if os.path.isdir(folder_name):
        # Remove the folder
        shutil.rmtree(folder_name)
        print(f"✅ {student_name}_{student_id} has been deleted.")
    else:
        print(f"❌ The student folder for {student_name}_{student_id} does not exist.")

if __name__ == "__main__":
    delete_student_folder()
