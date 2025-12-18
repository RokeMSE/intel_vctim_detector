FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    streamlit \
    ultralytics \
    opencv-python-headless \
    openvino \
    psutil \
    pandas \
    pillow

COPY main.py /app/main.py
COPY runs/detect/vctim_detector3/weights /app/weights

# Streamlit port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]