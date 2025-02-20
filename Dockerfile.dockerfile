# Use the official Python image as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install supervisord to manage multiple processes
RUN apt-get update && apt-get install -y supervisor && apt-get clean

# Expose ports for Chainlit and FastAPI
EXPOSE 8000 8001

# Copy the supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Run supervisord to manage both Chainlit and FastAPI
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]