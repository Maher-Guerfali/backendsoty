#!/bin/bash
# Exit on error
set -e

# Enable script debugging
set -x

# Use Render's PORT environment variable
PORT=${PORT:-8000}

# Log the port being used
echo "Starting server on port: $PORT"

# Install dependencies with verbose output
echo "Installing dependencies..."
pip install -r requirements.txt --no-cache-dir

# Create logs directory if it doesn't exist
echo "Creating logs directory..."
mkdir -p logs

# Set up environment for Render
echo "Setting up environment..."

# Ensure environment variables are loaded
echo "Loading environment variables..."
source .env 2>/dev/null || true

# Check required environment variables
echo "Checking required environment variables..."
if [ -z "$GROQ_API_KEY" ]; then
echo "Error: GROQ_API_KEY is not set"
exit 1
fi

if [ -z "$STABILITY_API_KEY" ]; then
echo "Error: STABILITY_API_KEY is not set"
exit 1
fi

# Set default values for optional variables
export FRONTEND_URL=${FRONTEND_URL:-https://mystoria-alpha.vercel.app}
export ENVIRONMENT=${ENVIRONMENT:-production}

# Start the server
echo "Starting server..."
uvicorn app.main:app --host 0.0.0.0 --port $PORT
export PYTHONUNBUFFERED=1
export PYTHONPATH=/opt/render/project/src:/app

# Run the FastAPI app with logging
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --log-level debug \
    --log-config uvicorn.json \
    --workers 1 \
    --access-log \
    --proxy-headers \
    --forwarded-allow-ips="*" \
    --root-path="/" \
    2>&1 | tee logs/app.log
