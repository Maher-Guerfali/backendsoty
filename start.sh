#!/usr/bin/env bash
# Exit on error
set -e

# Enable script debugging
set -x

# Use Render's PORT environment variable or default to 10000
export PORT=${PORT:-10000}

# Set host to 0.0.0.0 to make it accessible from outside the container
export HOST=0.0.0.0

# Make sure the script has execute permissions
chmod +x "$0"

# Log environment for debugging
echo "=== Environment Variables ==="
printenv | sort
echo "==========================="

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
[ -f ".env" ] && source .env

# Check required environment variables
echo "Checking required environment variables..."
if [ -z "$GROQ_API_KEY" ]; then
    echo "Error: GROQ_API_KEY is not set"
    exit 1
fi

if [ -z "$REPLICATE_API_TOKEN" ]; then
    echo "Error: REPLICATE_API_TOKEN is not set"
    exit 1
fi

# Set default values for optional variables
export FRONTEND_URL=${FRONTEND_URL:-https://mystoria-alpha.vercel.app}
export ENVIRONMENT=${ENVIRONMENT:-production}

# Debug: Show listening ports before starting
echo "=== Checking listening ports before start ==="
netstat -tuln || true
lsof -i :$PORT || true
echo "============================================"

# Start the server
echo "Starting FastAPI server on 0.0.0.0:$PORT..."

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
