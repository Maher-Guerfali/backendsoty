#!/bin/bash
# Exit on error
set -e

# Enable script debugging
set -x

# Check if required environment variables are set
if [ -z "$PORT" ]; then
    echo "Error: PORT environment variable is not set"
    exit 1
fi

# Install dependencies with verbose output
pip install -r requirements.txt --no-cache-dir

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the FastAPI app with logging
exec uvicorn app.main_new:app \
    --host 0.0.0.0 \
    --port $PORT \
    --log-level debug \
    --log-config uvicorn.json \
    2>&1 | tee logs/app.log
