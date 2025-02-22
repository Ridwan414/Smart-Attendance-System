# Smart Attendance System

A modern attendance tracking system that combines facial recognition and RFID card scanning capabilities for efficient and reliable attendance management.

## Features

### Facial Recognition Attendance
- Real-time face detection and recognition using MTCNN
- Support for multiple face capture during registration
- Automatic attendance marking with timestamp logging
- Visual progress tracking during face registration process
- Advanced face matching using face_recognition library

### RFID Card Scanning
- Quick and efficient attendance marking using RFID cards
- Streamlined card registration process
- Real-time card validation and verification
- Support for multiple RFID reader models

### User Management
- Dual registration options: face recognition and RFID
- Unique user ID system with data validation
- Comprehensive user profile management
- Automatic attendance history tracking

### Attendance Records
- Detailed daily attendance logs
- Export functionality to CSV format
- Real-time attendance status updates
- Historical attendance data visualization

## Prerequisites

- Python 3.8 or higher
- Webcam (for facial recognition features)
- RFID reader (optional, for card scanning capabilities)
- Internet connection for initial setup

## Installation

### Windows Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-attendance-system.git
cd smart-attendance-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Linux/MacOS Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-attendance-system.git
cd smart-attendance-system

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Starting the Application

1. Activate your virtual environment
2. Run the application:
   ```bash
   python src/main.py
   ```
3. Access the web interface at `http://localhost:5000`

### User Registration

1. Click "Add New User" in the navigation menu
2. Enter required user details
3. Choose registration method:
   - Facial Recognition: Follow the on-screen guide for face capture
   - RFID: Hold the card near the reader when prompted

### Marking Attendance

#### Facial Recognition
1. Select "Scan Faces" from the main menu
2. Position face(s) within the camera frame
3. System will automatically detect and mark attendance

#### RFID Scanning
1. Select "Scan ID Cards" from the main menu
2. Hold RFID card near the reader
3. Wait for confirmation beep

### Utilities

```bash
# Clear face recognition data
python src/utilities.py clear-face-data

# Reset attendance records
python src/utilities.py clear-attendance
```

## Key Dependencies

- OpenCV: Image processing and camera interface
- face_recognition: Core facial recognition functionality
- MTCNN: Face detection and alignment
- Flask: Web application framework
- pandas: Data handling and CSV operations
- Bootstrap: Frontend styling and components

## Best Practices

1. **Facial Recognition**
   - Ensure good lighting conditions
   - Maintain consistent face position during registration
   - Keep face within the guide box
   - Register multiple angles for better recognition

2. **RFID Scanning**
   - Hold cards steady near the reader
   - Avoid multiple cards in scanning range
   - Wait for confirmation before removing card

## Troubleshooting

- **Camera not detected**: Verify webcam connections and permissions
- **RFID reader issues**: Check USB connection and driver installation
- **Recognition problems**: Ensure proper lighting and face positioning
- **Database errors**: Verify write permissions in the data directory

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request