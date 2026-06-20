# Use an official Python image as a base image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and to prevent buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    scikit-optimize \
    torch \
    torchvision \
    seaborn \
    fastapi \
    uvicorn \
    jupyterlab \
    tensorflow \
    python-multipart \
    torchsummary \
    mlflow>=2.10 \
    "evidently>=0.7"

# Expose Jupyter, FastAPI, MLflow UI, model serving, and Evidently UI ports
EXPOSE 8000 8888 5050 5001 8080

# Default command to run interactive bash
CMD ["bash"]
