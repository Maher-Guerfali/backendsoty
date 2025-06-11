import os
import logging
import time
import base64
import io
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from PIL import Image
import requests

# Import routers
from app.api.endpoints.story import router as story_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get port from environment variable or default to 8000
PORT = int(os.getenv("PORT", 8000))
logger.info(f"Starting server on port: {PORT}")

# Ensure the host is set to 0.0.0.0 for Render.com
HOST = "0.0.0.0"

# Create FastAPI app
app = FastAPI(
    title="Story Generator API",
    description="API for generating children's stories with AI and image processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Pydantic models for image processing
class ImageToImageRequest(BaseModel):
    image: str  # base64 encoded image
    prompt: str
    strength: Optional[float] = 0.8
    guidance_scale: Optional[float] = 7.5
    num_inference_steps: Optional[int] = 20

class TextToImageRequest(BaseModel):
    prompt: str
    width: Optional[int] = 512
    height: Optional[int] = 512
    guidance_scale: Optional[float] = 7.5
    num_inference_steps: Optional[int] = 20

class ImageResponse(BaseModel):
    image: str  # base64 encoded result
    prompt_used: str
    processing_time: float

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
                "gemini_api_key": bool(os.getenv("AIzaSyAJcXKOOdp3pFfgvLGTU86YiQR3yIhpDdY")),
                "openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
                "replicate_api_key": bool(os.getenv("REPLICATE_API_TOKEN"))
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

def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def decode_base64_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

async def process_image_with_openai(image: Image.Image, prompt: str) -> str:
    """Process image using OpenAI DALL-E 2 image editing"""
    try:
        import openai
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        # Ensure image is in RGBA mode for editing
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Use DALL-E 2 for image editing (variation)
        response = client.images.create_variation(
            image=img_byte_arr.getvalue(),
            n=1,
            size="512x512"
        )
        
        # Download the generated image
        image_url = response.data[0].url
        img_response = requests.get(image_url)
        img_response.raise_for_status()
        
        # Convert to base64
        result_image = Image.open(io.BytesIO(img_response.content))
        return encode_image_to_base64(result_image)
        
    except Exception as e:
        logger.error(f"OpenAI image processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI processing failed: {str(e)}")

async def process_image_with_replicate(image: Image.Image, prompt: str, strength: float = 0.8) -> str:
    """Process image using Replicate's Stable Diffusion img2img"""
    try:
        import replicate
        
        replicate_token = os.getenv("REPLICATE_API_TOKEN")
        if not replicate_token:
            raise HTTPException(status_code=500, detail="Replicate API token not configured")
        
        # Convert image to base64 data URL
        image_b64 = encode_image_to_base64(image)
        
        # Use Stable Diffusion img2img model
        output = replicate.run(
            "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
            input={
                "image": image_b64,
                "prompt": prompt,
                "strength": strength,
                "guidance_scale": 7.5,
                "num_inference_steps": 20
            }
        )
        
        # Download the result
        if isinstance(output, list) and len(output) > 0:
            result_url = output[0]
        else:
            result_url = output
            
        img_response = requests.get(result_url)
        img_response.raise_for_status()
        
        result_image = Image.open(io.BytesIO(img_response.content))
        return encode_image_to_base64(result_image)
        
    except Exception as e:
        logger.error(f"Replicate image processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Replicate processing failed: {str(e)}")

async def generate_text_to_image_free(prompt: str) -> str:
    """Generate image from text using free service (Pollinations AI)"""
    try:
        # Use Pollinations AI as a free alternative
        base_url = "https://image.pollinations.ai/prompt/"
        encoded_prompt = requests.utils.quote(prompt)
        image_url = f"{base_url}{encoded_prompt}?width=512&height=512&nologo=true"
        
        response = requests.get(image_url)
        response.raise_for_status()
        
        result_image = Image.open(io.BytesIO(response.content))
        return encode_image_to_base64(result_image)
        
    except Exception as e:
        logger.error(f"Free text-to-image generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.post("/api/v1/image-to-image", response_model=ImageResponse)
async def image_to_image(request: ImageToImageRequest):
    """
    Transform an image based on a text prompt using AI image-to-image generation
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing image-to-image request with prompt: {request.prompt[:100]}...")
        
        # Decode the input image
        input_image = decode_base64_image(request.image)
        
        # Try different services in order of preference
        result_image_b64 = None
        
        # First try Replicate (if API key available)
        if os.getenv("REPLICATE_API_TOKEN"):
            try:
                result_image_b64 = await process_image_with_replicate(
                    input_image, 
                    request.prompt, 
                    request.strength
                )
                logger.info("Successfully processed with Replicate")
            except Exception as e:
                logger.warning(f"Replicate failed: {str(e)}")
        
        # Fallback to OpenAI (if API key available)
        if not result_image_b64 and os.getenv("OPENAI_API_KEY"):
            try:
                result_image_b64 = await process_image_with_openai(input_image, request.prompt)
                logger.info("Successfully processed with OpenAI")
            except Exception as e:
                logger.warning(f"OpenAI failed: {str(e)}")
        
        # If all paid services fail, create a simple text-to-image as fallback
        if not result_image_b64:
            logger.info("Using free text-to-image service as fallback")
            result_image_b64 = await generate_text_to_image_free(request.prompt)
        
        processing_time = time.time() - start_time
        
        return ImageResponse(
            image=result_image_b64,
            prompt_used=request.prompt,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Image-to-image processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {str(e)}"
        )

@app.post("/api/v1/text-to-image", response_model=ImageResponse)
async def text_to_image(request: TextToImageRequest):
    """
    Generate an image from a text prompt
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing text-to-image request with prompt: {request.prompt[:100]}...")
        
        # Use free service for text-to-image
        result_image_b64 = await generate_text_to_image_free(request.prompt)
        
        processing_time = time.time() - start_time
        
        return ImageResponse(
            image=result_image_b64,
            prompt_used=request.prompt,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Text-to-image processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate image: {str(e)}"
        )

@app.post("/api/v1/upload-and-process")
async def upload_and_process_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    strength: float = Form(0.8)
):
    """
    Upload an image file and process it with a prompt
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to base64
        image_b64 = encode_image_to_base64(image)
        
        # Process with image-to-image
        request = ImageToImageRequest(
            image=image_b64,
            prompt=prompt,
            strength=strength
        )
        
        return await image_to_image(request)
        
    except Exception as e:
        logger.error(f"Upload and process failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload and process image: {str(e)}"
        )

# Include routers
app.include_router(story_router, prefix="/api/v1", tags=["stories"])

# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8080",
    "https://localhost:3000",
    "https://localhost:8000",
    "https://mystoria-alpha.vercel.app",
    "https://*.vercel.app",
    "https://*.vercel.com",
    "https://storiabackend.onrender.com",
    "https://*.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Authorization", "X-Requested-With", "Access-Control-Allow-Origin", "Access-Control-Allow-Credentials"],
    allow_origin_regex="https://.*\.vercel\.app|https://.*\.vercel\.com|https://.*\.onrender\.com",
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Include router after CORS is configured
app.include_router(story_router, prefix="/api/v1", tags=["stories"])

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
        host=HOST,
        port=PORT,
        reload=True
    )