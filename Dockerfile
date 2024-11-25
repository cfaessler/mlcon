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
    torch \
    torchvision \
    seaborn \
    fastapi \
    uvicorn \
    jupyterlab \
    tensorflow

# Expose FastAPI default port
EXPOSE 8000 8888

# Default command to run interactive bash
CMD ["bash"]
