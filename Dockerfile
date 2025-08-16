# 1. Use an official, slim Python base image
FROM python:3.11-slim

# 2. Set environment variables
# - PYTHONUNBUFFERED: Ensures logs are sent straight to the container's log stream
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy and install dependencies first (for better caching)
# This step is only re-run if requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code into the container
COPY . .

# 6. The command to run your application
# This is the replacement for the Procfile!
# It tells gunicorn to listen on all network interfaces (0.0.0.0)
# on the port provided by Railway's $PORT environment variable.
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:server"]
