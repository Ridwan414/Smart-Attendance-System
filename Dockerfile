# Use Python 3.8 as base image
FROM python:3.8-slim

# Install system dependencies required for OpenCV and face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    python3-dev \
    python3-pip \
    wget \
    libv4l-dev \ 
    libxvidcore-dev \
    libx264-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p static/faces Attendance

# Expose port 5000 for Flask
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=src/app.py
ENV DISPLAY=:0

# Run the application
CMD ["python", "src/app.py"]