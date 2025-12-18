# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
# 'libgl1' and 'libglib2.0-0' are critical for cv2 to run in Docker
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
# We write requirements inline to keep it simple for you
RUN pip install --no-cache-dir \
    streamlit \
    ultralytics \
    opencv-python-headless \
    openvino \
    psutil \
    pandas \
    pillow

# Copy the application code
COPY main.py /app/main.py

# Create a folder for weights and copy them
# NOTE: Ensure your local folder structure matches this
COPY runs/detect/vctim_detector3/weights /app/weights

# Expose Streamlit port
EXPOSE 8501

# Healthcheck to ensure container is alive
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]