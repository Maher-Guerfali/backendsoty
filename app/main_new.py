import os
import logging
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

# Import routers
from app.api.endpoints.story import router as story_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcode port to 8000
PORT = 8000
logger.info(f"Starting server on port: {PORT}")

# Create FastAPI app
app = FastAPI(
    title="Story Generator API",
    description="API for generating children's stories with AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add root health check endpoint
@app.get("/")
async def root():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Add detailed health check endpoint
@app.get("/health")
async def health_check():
    try:
        return {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "port": PORT,
            "environment": os.getenv("ENVIRONMENT", "production"),
            "dependencies": {
                "groq_api_key": bool(os.getenv("GROQ_API_KEY")),
                "stability_api_key": bool(os.getenv("STABILITY_API_KEY"))
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Include routers
app.include_router(story_router, prefix="/api/v1", tags=["stories"])

# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "https://localhost:3000",
    "https://localhost:8000",
    "https://mystoria-alpha.vercel.app",
    "https://*.vercel.app",
    "https://*.vercel.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Authorization", "X-Requested-With", "Access-Control-Allow-Origin", "Access-Control-Allow-Credentials"],
    allow_origin_regex="https://.*\.vercel\.app|https://.*\.vercel\.com",
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Include routers
app.include_router(story_router, prefix="/api/v1", tags=["stories"])

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Check basic service status
        service_status = "healthy"
        
        # Add more detailed health checks here
        # For example, check database connection if using one
        # Check API keys if available
        
        return {
            "status": service_status,
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "details": {
                "environment": os.getenv("ENVIRONMENT", "production"),
                "service": "story-generator",
                "dependencies": {
                    "fastapi": fastapi.__version__
                }
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.detail}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
            "request": {
                "method": request.method,
                "path": request.url.path,
                "headers": dict(request.headers)
            }
        },
    )

# Add middleware for logging requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(
            f"Request: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.2f}s"
        )
        
        return response
        
    except Exception as e:
        logger.error(
            f"Request failed: {request.method} {request.url.path} - "
            f"Error: {str(e)}",
            exc_info=True
        )
        raise

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
