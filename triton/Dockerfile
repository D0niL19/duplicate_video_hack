FROM nvcr.io/nvidia/tritonserver:24.08-py3

# Install ffmpeg with as few dependencies as possible
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir \
    opencv-python-headless \
    transformers \
    pillow \
    tritonclient[all] \
    ffmpeg-python \
    librosa \
    soundfile

EXPOSE 8000
