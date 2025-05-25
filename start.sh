#!/bin/bash
# Exit on error
set -e

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI app
uvicorn app.main:app --host 0.0.0.0 --port $PORT
