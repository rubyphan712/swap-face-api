# Use official Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0

# Set workdir
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose the port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
