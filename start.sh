#!/usr/bin/env bash
# Exit on error and print commands as they're executed
set -ex

# Set default values
export PORT=${PORT:-10000}
export HOST=0.0.0.0
export PYTHONUNBUFFERED=1

# Enable script debugging
echo "=== Starting server on port $PORT ==="

# Check for required commands
for cmd in python3 pip; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is not installed"
        exit 1
    fi
done

# Log environment for debugging
echo "=== Environment Variables ==="
printenv | sort
echo "==========================="

# Create logs directory
mkdir -p logs

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

# Load environment variables if .env exists
if [ -f ".env" ]; then
    echo "Loading .env file..."
    set -o allexport
    source .env
    set +o allexport
fi

# Check required environment variables
for var in GROQ_API_KEY REPLICATE_API_TOKEN; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not set"
        exit 1
    fi
done

# Set defaults for optional variables
export FRONTEND_URL=${FRONTEND_URL:-https://mystoria-alpha.vercel.app}
export ENVIRONMENT=${ENVIRONMENT:-production}

# Debug info
echo "=== System Info ==="
python3 --version
pip --version
echo "Current directory: $(pwd)"
echo "Files in current directory:"
ls -la
echo "==================="

# Check if uvicorn.json exists
if [ ! -f "uvicorn.json" ]; then
    echo "Error: uvicorn.json not found in $(pwd)"
    exit 1
fi

# Start the server
echo "Starting Uvicorn..."
exec uvicorn app.main:app \
    --host $HOST \
    --port $PORT \
    --workers 1 \
    --log-level info \
    --log-config uvicorn.json \
    --access-log \
    --proxy-headers \
    --timeout-keep-alive 30 \
    --no-server-header \
    2>&1 | tee -a logs/app.log
