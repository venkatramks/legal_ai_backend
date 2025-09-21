# Use Python 3.11 slim as the base
FROM python:3.11-slim

# Install system packages required for OCR (poppler and tesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY backend/ /app/

# Expose port
ENV PORT 8080

# Start the app
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8080", "-w", "2", "--threads", "4"]
