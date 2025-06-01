#!/bin/bash
# Exit on error
set -e

# Enable script debugging
set -x

# Set a hardcoded port
PORT=8000

# Log the port being used
echo "Starting server on port: $PORT"

# Install dependencies with verbose output
pip install -r requirements.txt --no-cache-dir

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up environment for Render
echo "Setting up environment for Render"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/opt/render/project/src
export PORT

# Run the FastAPI app with logging
exec uvicorn app.main_new:app \
    --host 0.0.0.0 \
    --port $PORT \
    --log-level debug \
    --log-config uvicorn.json \
    --reload \
    --workers 1 \
    --access-log \
    --proxy-headers \
    --forwarded-allow-ips="*" \
    --root-path="/" \
    2>&1 | tee logs/app.log
