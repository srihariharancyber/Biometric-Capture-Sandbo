FROM python:3.11-slim

# Prevent Python from writing .pyc and enable buffer flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

# Install build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and pre-trained weights
COPY . .

# Expose target port
EXPOSE 5000

# Start production WSGI server
CMD exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120
