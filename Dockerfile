# Builder stage
FROM python:3.8-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    python3-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set up virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.8-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libx11-6 \
    libgtk-3-0 \
    libboost-python-dev \
    libv4l-0 \
    libxvidcore4 \
    libx264-dev \
    ffmpeg \
    libqt5gui5 \
    libqt5core5a \
    libqt5widgets5 \
    libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p static/faces Attendance

# Copy application code
COPY src/ ./src/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    FLASK_APP=src/app.py \
    DISPLAY=:0

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "src/app.py"]