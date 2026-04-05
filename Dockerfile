FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# ffmpeg is essential for Whisper and audio preprocessing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for persistent storage and models
RUN mkdir -p user_data pretrained_models

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Use Gunicorn as the production WSGI server
# Limit workers to 1 to prevent Out-Of-Memory errors with Whisper/PyTorch
# Increase timeout because transcription can take time
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "300", "server:app"]
