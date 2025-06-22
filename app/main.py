import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Import routers
from app.api.endpoints.story import router as story_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get port from environment variable or default to 10000
PORT = int(os.getenv("PORT", 10000))
HOST = os.getenv("HOST", "0.0.0.0")

# Log the host and port being used
logger.info(f"Starting server on {HOST}:{PORT}")

# Initialize FastAPI app
app = FastAPI(
    title="Story Generator API",
    description="API for generating children's stories with AI",
    version="0.1.0"
)

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Story Generator API. Use /docs for the API documentation."}

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint that verifies the API is running
    """
    return {
        "status": "healthy",
        "version": "0.1.0"
    }

# Include routers
app.include_router(story_router, prefix="/api/v1", tags=["stories"])

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"},
    )

# This block is only for local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=True
    )

# For production on Render.com, the app will be started using:
# uvicorn app.main:app --host 0.0.0.0 --port $PORT

# This is needed for production on Render
app.state.host = HOST
app.state.port = PORT