FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires a non-root user for security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements
COPY --chown=user requirements.txt .

# Upgrade pip and install CPU torch before requirements to save space and time
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Create directories for persistent storage
RUN mkdir -p user_data pretrained_models

# Hugging Face runs traffic on port 7860
EXPOSE 7860

# We bind gunicorn to port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "300", "server:app"]
