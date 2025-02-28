import os
import glob
from datetime import datetime

def clear_attendance():
    # Ask for the option to clear by day or month
    print("Select an option to clear attendance:")
    print("1. Clear attendance by specific day")
    print("2. Clear attendance by specific month")
    
    choice = input("Enter 1 or 2: ")
    
    if choice == '1':
        # Clear by specific day
        date_input = input("Enter the date (format: DD-MM-YYYY): ")
        try:
            # Validate the input date
            date = datetime.strptime(date_input, '%d-%m-%Y')
            filename = f"Attendance/Attendance-{date.strftime('%d_%m_%Y')}.xlsx"
            
            # Check if the file exists and delete it
            if os.path.exists(filename):
                os.remove(filename)
                print(f"✅ Attendance for {date.strftime('%d-%B-%Y')} has been cleared.")
            else:
                print(f"❌ No attendance file found for {date.strftime('%d-%B-%Y')}.")
        except ValueError:
            print("❌ Invalid date format. Please use DD-MM-YYYY.")
    
    elif choice == '2':
        # Clear by specific month
        month_input = input("Enter the month and year (format: MM-YYYY): ")
        try:
            # Validate the input month and year
            month_year = datetime.strptime(month_input, '%m-%Y')
            # Create a pattern for searching all files for that month
            pattern = f"Attendance/Attendance-*_{month_year.strftime('%m_%Y')}.xlsx"
            files = glob.glob(pattern)
            
            if files:
                for file in files:
                    os.remove(file)
                    print(f"✅ Cleared attendance for {os.path.basename(file)}.")
            else:
                print(f"❌ No attendance files found for {month_year.strftime('%B %Y')}.")
        except ValueError:
            print("❌ Invalid month format. Please use MM-YYYY.")
    
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    clear_attendance()
