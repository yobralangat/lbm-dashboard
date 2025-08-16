# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy your app code
COPY . /app
COPY requirements.txt .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Railway will use
EXPOSE 8000

# Start the app with Gunicorn
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8000"]

ENV TRANSACTIONS_URL=https://docs.google.com/spreadsheets/d/e/2PACX-1vSSVhY9V9Pln14qWv1oQS3e-mNoyyYyiXfym1CHw-luAIrbP3Zg2EqnPXGVxoDJQQDxTVP5Es9EdWUW/pub?gid=0&single=true&output=csv
ENV PRODUCTS_URL=https://docs.google.com/spreadsheets/d/e/2PACX-1vQE2fmwEbJKVkU701qiYvY2MgZxb_C74DDluLbIDgrzz1HZ8bUO4BFXCUhQh-tdiVunBB81_giASCAC/pub?gid=1791698933&single=true&output=csv
