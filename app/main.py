from typing import List, Optional, Dict, Any, Union
import base64
import io
import os
import sys
import traceback
import uuid
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, Field, Json, HttpUrl
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import asyncio
import replicate
import time
import httpx
import json
import tempfile
import numpy as np
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# In-memory storage for story state (in a real app, use a database)
story_states = {}

# WebSocket connections for real-time updates
active_connections = {}

# Story store for tracking story generation
STORY_STORE: Dict[str, dict] = {}

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, story_id: str, websocket: WebSocket):
        await websocket.accept()
        if story_id not in self.active_connections:
            self.active_connections[story_id] = []
        self.active_connections[story_id].append(websocket)

    def disconnect(self, story_id: str, websocket: WebSocket):
        if story_id in self.active_connections:
            self.active_connections[story_id].remove(websocket)

    async def broadcast(self, story_id: str, message: dict):
        if story_id in self.active_connections:
            for connection in self.active_connections[story_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"Error sending WebSocket message: {e}")

manager = ConnectionManager()

# Set numpy to ignore some warnings
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Stability AI Configuration
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')
STABILITY_API_HOST = 'https://api.stability.ai'
STABILITY_ENGINE_ID = 'stable-diffusion-xl-1024-v1-0'  # or 'stable-diffusion-v1-6' for older models

# Get allowed origins from environment variable or use default
FRONTEND_URL = os.getenv('FRONTEND_URL', 'https://mystoria-alpha.vercel.app')

app = FastAPI(title="Pirate Story Generator API")

# CORS middleware configuration
# In development, allow all origins for easier debugging
# In production, you should restrict this to your frontend domain
is_production = os.getenv('ENVIRONMENT', 'development') == 'production'

# Base allowed origins
allowed_origins = []

# Add frontend URL if specified
if FRONTEND_URL:
    # Add both with and without trailing slash
    allowed_origins.extend([
        FRONTEND_URL.rstrip('/'),
        f"{FRONTEND_URL.rstrip('/')}/"
    ])

# Development origins (only add if not in production)
if not is_production:
    development_origins = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://localhost:3000",
        "https://localhost:8000",
        "https://localhost:5173",
        "https://localhost:5174",
        "https://127.0.0.1:3000",
        "https://127.0.0.1:8000",
        "https://127.0.0.1:5173",
        "https://127.0.0.1:5174",
    ]
    allowed_origins.extend(development_origins)

# Remove duplicates
allowed_origins = list(set(allowed_origins))

# Log allowed origins for debugging
print("\n" + "="*50)
print(f"Environment: {'PRODUCTION' if is_production else 'DEVELOPMENT'}")
print(f"Frontend URL: {FRONTEND_URL}")
print(f"Allowed Origins: {json.dumps(allowed_origins, indent=2)}")
print("="*50 + "\n")

# Add CORS middleware with WebSocket support
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r'https?://.*\.?vercel\.app/?',  # Allow any Vercel deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add WebSocket origin checking middleware
@app.middleware("http")
async def check_origin(request: Request, call_next):
    # Skip origin check for WebSocket upgrade requests
    if "upgrade" in request.headers.get("connection", "").lower():
        return await call_next(request)
        
    # For regular HTTP requests, use CORS middleware
    return await call_next(request)

# API settings
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')
REPLICATE_API_KEY = os.getenv('REPLICATE_API_KEY')

# Configure Replicate
if REPLICATE_API_KEY:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY

class StoryPart(BaseModel):
    text: str
    image_url: Optional[str] = None
    image_prompt: Optional[str] = None
    image_status: str = "pending"  # pending, processing, completed, failed
    error: Optional[str] = None

class StoryRequest(BaseModel):
    child_name: str = ""
    theme: str = ""
    face_image: Optional[str] = None  # Base64 encoded face image



class StoryResponse(BaseModel):
    title: str
    parts: List[StoryPart]

def generate_pirate_story_with_groq(child_name: str, theme: str):
    """Generate an 8-page pirate story using Groq API."""
    # Generate pirate story using Groq API
    
    prompt = f"""Create an 8-page children's pirate adventure story about {child_name}. 

IMPORTANT: {child_name} is a young pirate child (8-12 years old) who will be the main character in every page.

Each page should be a complete mini-adventure that connects to the overall story but can be understood on its own.

Format your response as valid JSON like this:
{{
    "title": "Captain {child_name}'s [Adventure Name]",
    "pages": [
        {{
            "page_number": 1,
            "story_text": "A full paragraph (4-5 sentences) describing this part of the adventure...",
            "page_summary": "Brief 1-sentence summary of what happens on this page",
            "image_prompt": "Detailed visual description for this page featuring young pirate {child_name}"
        }}
    ]
}}

Story requirements:
- Each page should have 4-5 sentences of story text
- Each page should work as a mini-adventure but connect to the overall story
- Focus on themes like: treasure hunting, sea adventures, making friends, solving puzzles, helping others
- Keep it positive and age-appropriate for children 6-12
- Make {child_name} the brave hero of each page

For image_prompt, describe the scene but don't mention the art style - that will be added automatically.
Focus on: {child_name}'s actions, the environment, other characters, objects in the scene."""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 3000
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        result = response.json()
        
        if 'error' in result:
            pass
            return None
            
        if 'choices' not in result or not result['choices']:
            return None
            
        content = result['choices'][0]['message']['content']
        
        try:
            # Try to parse as JSON
            story_data = json.loads(content)
            return story_data
        except json.JSONDecodeError:
            # Extract JSON from content if wrapped in other text
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                story_data = json.loads(json_str)
                return story_data
            else:
                print("Could not extract JSON from response")
                return None
                    
    except Exception as e:
        print(f"Groq API error: {e}")
        return None

def create_flapjack_style_prompt(base_prompt: str, child_name: str) -> str:
    """Enhance the image prompt with Flapjack cartoon style and old paper aesthetics."""
    
    style_prompt = f"""
    {base_prompt}

    Art style: The Marvelous Misadventures of Flapjack cartoon style, 2D animation look, 
    cartoonish and whimsical. {child_name} is a young pirate child (8-12 years old) with 
    the same face throughout all images.

    Visual aesthetics: Old yellowed paper texture background, vintage comic book style, 
    aged parchment look with slightly torn edges, sepia and warm yellow tones, 
    hand-drawn illustration feel like an old children's book.

    Character consistency: {child_name} always appears as the same young pirate child 
    with consistent facial features, pirate outfit (bandana, vest, boots), 
    adventurous and cheerful expression.

    Overall mood: Whimsical, adventurous, child-friendly, colorful but with vintage 
    paper texture overlay.
    """
    
    return style_prompt.strip()

def fix_base64_padding(b64_string: str) -> str:
    """Ensure base64 string has correct padding."""
    # Remove data URL prefix if present
    if b64_string.startswith('data:'):
        b64_string = b64_string.split(',', 1)[1]
    
    # Add padding if needed
    padding = len(b64_string) % 4
    if padding:
        b64_string += '=' * (4 - padding)
    
    return b64_string

async def generate_image_with_replicate(
    prompt: str, 
    child_name: str, 
    face_image_b64: Optional[str] = None,
    strength: float = 0.7,
    width: int = 1024,
    height: int = 1024
) -> Optional[Dict[str, Any]]:
    """
    Generate an image using Replicate API asynchronously.
    
    Args:
        prompt: Text prompt for image generation
        child_name: Name of the child (used for prompt enhancement)
        face_image_b64: Optional base64-encoded image for image-to-image
        strength: How much to transform the input image (0.0 to 1.0)
        width: Width of the output image
        height: Height of the output image
    """
    if not REPLICATE_API_KEY:
        return {"error": "Replicate API key not configured"}
        
    temp_path = None
    try:
        # Enhance the prompt with Flapjack style
        enhanced_prompt = create_flapjack_style_prompt(prompt, child_name)
        print(f"\n=== Generating image with Replicate ===")
        print(f"Prompt: {enhanced_prompt}")
        
        # Prepare input parameters
        input_params = {
            "prompt": enhanced_prompt,
            "width": width,
            "height": height,
            "num_outputs": 1,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "scheduler": "K_EULER",
            "seed": None,
            "negative_prompt": "blurry, low quality, distorted, disfigured, extra limbs, extra fingers, cropped, out of frame, watermark, signature, text"
        }
        
        # Add image-to-image parameters if input image is provided
        if face_image_b64:
            # Save the base64 image to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                # Remove data URL prefix if present
                image_data = base64.b64decode(face_image_b64.split(",")[-1])
                temp_file.write(image_data)
                temp_path = temp_file.name
            
            try:
                # Open the image and ensure it's in RGB mode
                with Image.open(temp_path) as img:
                    img = img.convert("RGB")
                    img.save(temp_path, "PNG")
                
                # Update parameters for image-to-image
                input_params.update({
                    "image": open(temp_path, "rb"),
                    "prompt_strength": strength,
                    "num_inference_steps": 75,  # More steps for better quality with img2img
                })
                
                print(f"Using image-to-image with strength: {strength}")
                
                # Call the Replicate API with image-to-image
                output = await asyncio.to_thread(
                    replicate.run,
                    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                    input=input_params
                )
                
                # Process the output URL
                if not output or not isinstance(output, list) or not output[0]:
                    return {"error": "Unexpected output format from Replicate"}
                
                image_url = output[0]
                response = requests.get(image_url)
                if response.status_code != 200:
                    return {"error": f"Failed to download generated image: {response.status_code}"}
                
                # Convert to base64
                image_base64 = base64.b64encode(response.content).decode("utf-8")
                return {"image": f"data:image/png;base64,{image_base64}", "source": "replicate-sdxl"}
                
            finally:
                # Clean up the temporary file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
        else:
            # Text-to-image generation
            print("Using text-to-image generation")
            output = await asyncio.to_thread(
                replicate.run,
                "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input=input_params
            )
            
            # Process the output URL
            if not output or not isinstance(output, list) or not output[0]:
                return {"error": "Unexpected output format from Replicate"}
            
            image_url = output[0]
            response = requests.get(image_url)
            if response.status_code != 200:
                return {"error": f"Failed to download generated image: {response.status_code}"}
            
            # Convert to base64
            image_base64 = base64.b64encode(response.content).decode("utf-8")
            return {"image": f"data:image/png;base64,{image_base64}", "source": "replicate-sdxl"}
            
    except Exception as e:
        error_msg = f"Error with Replicate: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        return {"error": error_msg}
        
    finally:
        # Ensure temporary file is cleaned up
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Error cleaning up temporary file: {str(e)}")

async def generate_image(
    prompt: str, 
    child_name: str, 
    face_image_b64: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    strength: float = 0.7
) -> Dict[str, str]:
    """
    Generate image using available services with detailed error reporting.
    
    Args:
        prompt: Text prompt for image generation
        child_name: Name of the child (for prompt enhancement)
        face_image_b64: Optional base64-encoded image for image-to-image
        width: Width of the output image (default: 1024)
        height: Height of the output image (default: 1024)
        strength: How much to transform the input image (0.0 to 1.0, default: 0.7)
        
    Returns:
        Dictionary with 'image' (base64) or 'error' key
    """
    print(f"\n=== Starting image generation ===")
    print(f"Prompt: {prompt}")
    if face_image_b64:
        print("Using image-to-image with provided face")
    
    # Try different services in order of preference
    services = []
    
    # Add Stability AI if API key is available
    if STABILITY_API_KEY:
        services.append(("stability-ai-sd3", _try_stability_ai, {
            "prompt": prompt, 
            "child_name": child_name,
            "face_image_b64": face_image_b64,
            "width": width,
            "height": height,
            "strength": strength
        }))
    
    # Add Replicate if API key is available
    if REPLICATE_API_KEY:
        services.append(("replicate", generate_image_with_replicate, {
            "prompt": prompt,
            "child_name": child_name,
            "face_image_b64": face_image_b64,
            "width": width,
            "height": height,
            "strength": strength
        }))
    
    if not services:
        error_msg = "No image generation services are properly configured. Please check your API keys."
        print(error_msg)
        return {"error": error_msg, "attempts": []}
    
    errors = []
    
    for service_name, service_func, kwargs in services:
        print(f"\nTrying {service_name}...")
        try:
            result = await service_func(**kwargs)
            
            if isinstance(result, dict):
                if "image" in result:
                    print(f"Successfully generated image with {service_name}")
                    return result
                elif "error" in result:
                    error_msg = f"{service_name} error: {result['error']}"
                    print(error_msg)
                    errors.append({"service": service_name, "error": error_msg})
                else:
                    error_msg = f"Unexpected response format from {service_name}"
                    print(error_msg)
                    errors.append({"service": service_name, "error": error_msg})
            else:
                error_msg = f"Unexpected return type from {service_name}: {type(result)}"
                print(error_msg)
                errors.append({"service": service_name, "error": error_msg})
                
        except Exception as e:
            error_msg = f"Error with {service_name}: {str(e)}"
            print(f"{error_msg}\n{traceback.format_exc()}")
            errors.append({"service": service_name, "error": str(e)})
    
    # If we get here, all services failed
    error_details = "\n".join([f"- {e['service']}: {e['error']}" for e in errors])
    final_error = f"All image generation attempts failed. Details:\n{error_details}"
    print(f"\n=== Image Generation Failed ===\n{final_error}")
    return {"error": final_error, "attempts": errors}

async def _try_stability_ai(
    prompt: str, 
    face_image_b64: Optional[str], 
    child_name: str,
    width: int = 1024,
    height: int = 1024,
    strength: float = 0.7
) -> Dict[str, str]:
    """
    Generate or transform an image using Stability AI's API.
    
    Args:
        prompt: Text prompt for image generation
        face_image_b64: Optional base64-encoded image for image-to-image
        child_name: Name of the child (for prompt enhancement)
        width: Width of the output image (default: 1024)
        height: Height of the output image (default: 1024)
        strength: How much to transform the input image (0.0 to 1.0, default: 0.7)
        
    Returns:
        Dictionary with 'image' (base64) or 'error' key
    """
    if not STABILITY_API_KEY:
        return {"error": "Stability API key not configured"}
    
    # Enhance the prompt with Flapjack style
    enhanced_prompt = create_flapjack_style_prompt(prompt, child_name)
    print(f"\n=== Generating image with Stability AI ===")
    print(f"Prompt: {enhanced_prompt}")
    
    try:
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Accept": "application/json"
        }
        
        # Prepare the request data
        data = {
            "prompt": enhanced_prompt,
            "output_format": "png",
            "width": width,
            "height": height,
            "samples": 1,
            "steps": 30,
        }
        
        files = {"none": ''}  # Required by the API even if not used
        
        # If we have an input image, add it to the request
        if face_image_b64:
            print("Using image-to-image transformation")
            try:
                # Remove data URL prefix if present
                if "," in face_image_b64:
                    face_image_b64 = face_image_b64.split(",")[1]
                    
                # Convert base64 to bytes
                image_data = base64.b64decode(face_image_b64)
                
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_file.write(image_data)
                    temp_path = temp_file.name
                
                try:
                    # Open the image and ensure it's in RGB mode
                    with Image.open(temp_path) as img:
                        img = img.convert("RGB")
                        img_io = io.BytesIO()
                        img.save(img_io, format='PNG')
                        img_io.seek(0)
                        
                        files["image"] = ("input.png", img_io, "image/png")
                        data["mode"] = "image-to-image"
                        data["strength"] = strength
                        
                        # Make the API request
                        response = requests.post(
                            f"{STABILITY_API_HOST}/v2beta/stable-image/generate/sd3",
                            headers=headers,
                            files=files,
                            data=data,
                            timeout=60
                        )
                finally:
                    # Clean up the temporary file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            except Exception as e:
                error_msg = f"Error processing face image: {str(e)}"
                print(error_msg)
                return {"error": error_msg}
        else:
            # Text-to-image generation
            print("Using text-to-image generation")
            try:
                response = requests.post(
                    f"{STABILITY_API_HOST}/v2beta/stable-image/generate/sd3",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=60
                )
            except requests.exceptions.RequestException as e:
                error_msg = f"Request to Stability AI API failed: {str(e)}"
                print(error_msg)
                return {"error": error_msg}
        
        # Check for errors
        if response.status_code != 200:
            error_msg = f"Stability API error: {response.status_code} - {response.text}"
            print(error_msg)
            return {"error": error_msg}
        
        # Return the image as base64
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        return {"image": f"data:image/png;base64,{image_base64}", "source": "stability-ai-sd3"}
        
    except Exception as e:
        error_msg = f"Error with Stability AI: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        return {"error": error_msg}

def create_fallback_pirate_story(child_name: str):
    """Fallback pirate story when API fails."""
    return {
        "title": f"Captain {child_name}'s Treasure Island Adventure",
        "pages": [
            {
                "page_number": 1,
                "story_text": f"Young {child_name} discovered an ancient treasure map hidden in their grandmother's attic. The map was drawn on yellowed parchment and showed a mysterious island with an X marking buried treasure. {child_name} decided this was the perfect day for a grand pirate adventure. They quickly put on their favorite pirate hat and grabbed their wooden sword.",
                "page_summary": f"{child_name} finds a treasure map and decides to become a pirate",
                "image_prompt": f"Young pirate {child_name} in attic holding old treasure map, wearing pirate hat and vest, excited expression, dusty attic with old trunks and spider webs"
            },
            {
                "page_number": 2,
                "story_text": f"Captain {child_name} built a magnificent pirate ship from cardboard boxes and set sail across the backyard ocean. The ship had a tall mast with a black flag featuring a friendly skull. As they sailed, {child_name} sang pirate songs and watched for sea monsters. The wind filled their imagination with dreams of adventure and treasure.",
                "page_summary": f"Captain {child_name} sails away on their cardboard pirate ship",
                "image_prompt": f"Pirate captain {child_name} standing on cardboard ship deck in backyard, pirate flag flying, wooden sword at side, looking out at imaginary ocean"
            },
            {
                "page_number": 3,
                "story_text": f"The mysterious treasure island appeared on the horizon, covered with tall palm trees and golden beaches. {child_name} carefully navigated through dangerous waters filled with friendly rubber ducky sea creatures. As they approached the shore, colorful parrots welcomed them with cheerful squawks. The island looked exactly like the one on the treasure map.",
                "page_summary": f"{child_name} reaches the mysterious treasure island",
                "image_prompt": f"Pirate {child_name} approaching tropical island on ship, palm trees and golden beach visible, colorful parrots flying around, rubber ducky sea creatures in water"
            },
            {
                "page_number": 4,
                "story_text": f"Following the treasure map through the jungle, {child_name} discovered a hidden cave behind a waterfall. The cave entrance sparkled with colorful crystals and glowed with mysterious light. {child_name} bravely entered the cave, using their flashlight to guide the way. Strange but friendly cave creatures watched from the shadows, curious about this young pirate explorer.",
                "page_summary": f"{child_name} explores a magical crystal cave behind a waterfall",
                "image_prompt": f"Brave pirate {child_name} entering crystal cave behind waterfall, flashlight in hand, colorful crystals on cave walls, friendly creatures peeking from shadows"
            },
            {
                "page_number": 5,
                "story_text": f"Deep inside the cave, {child_name} met Captain Silverbeard, a friendly ghost pirate who had been guarding the treasure for a hundred years. Captain Silverbeard told amazing stories of sea adventures and pirate legends. He was impressed by {child_name}'s courage and kindness. The ghost captain decided that {child_name} was worthy of learning the treasure's secret.",
                "page_summary": f"{child_name} meets the friendly ghost pirate Captain Silverbeard",
                "image_prompt": f"Young pirate {child_name} talking with friendly ghost pirate Captain Silverbeard in glowing cave, both wearing pirate outfits, treasure chests visible in background"
            },
            {
                "page_number": 6,
                "story_text": f"Captain Silverbeard revealed that the real treasure wasn't gold or jewels, but a magical compass that always points toward adventure and friendship. The compass glowed with warm light and hummed with ancient magic. {child_name} learned that the greatest treasures are the friends you make and the adventures you share. The ghost captain smiled proudly as he passed on this wisdom.",
                "page_summary": f"{child_name} receives a magical compass and learns about true treasure",
                "image_prompt": f"Ghost pirate giving glowing magical compass to young pirate {child_name}, warm golden light surrounding them, ancient treasure cave setting"
            },
            {
                "page_number": 7,
                "story_text": f"Using the magical compass, {child_name} helped rescue a family of lost sea turtles who couldn't find their way home. The compass led them through coral reefs and past sleeping sea dragons to the turtles' underwater city. The grateful turtle family invited {child_name} to visit anytime. {child_name} realized that helping others felt better than finding any treasure.",
                "page_summary": f"{child_name} uses the magical compass to help lost sea turtles",
                "image_prompt": f"Pirate {child_name} swimming underwater with sea turtle family, magical compass glowing, underwater city with coral reefs and sea dragons in background"
            },
            {
                "page_number": 8,
                "story_text": f"Captain {child_name} sailed home as the sun set over the ocean, painting the sky in brilliant oranges and purples. Back in their backyard, they carefully put away their pirate hat and treasure map, but kept the magical compass close. {child_name} fell asleep that night dreaming of tomorrow's adventures, knowing that the greatest treasure is a heart full of courage and kindness.",
                "page_summary": f"Captain {child_name} returns home with memories and the magical compass",
                "image_prompt": f"Pirate {child_name} back home at sunset, holding magical compass, pirate ship in background, warm orange and purple sky, peaceful end to adventure"
            }
        ]
    }

@app.post("/upload-face")
async def upload_face(file: UploadFile = File(...)):
    """Upload a face image and return its base64 encoded content."""
    try:
        # Read the file content
        contents = await file.read()
        
        # Encode to base64
        base64_encoded = base64.b64encode(contents).decode('utf-8')
        
        return {"face_image": base64_encoded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

class PirateImageRequest(BaseModel):
    face_image: str  # Base64 encoded image
    prompt: str
    child_name: str

@app.post("/api/generate-pirate-image", response_model=Dict[str, str])
async def generate_pirate_image(request: PirateImageRequest):
    """Test endpoint to generate a pirate image with a face."""
    try:
        print("\n=== Starting pirate image generation ===")
        print(f"Request received with prompt: {request.prompt[:100]}...")
        print(f"Child name: {request.child_name}")
        print(f"Face image provided: {'Yes' if request.face_image else 'No'}")
        
        # Check if we have the required API keys
        if not STABILITY_API_KEY and not os.getenv('REPLICATE_API_TOKEN'):
            error_msg = "No API keys found. Please set both STABILITY_API_KEY and REPLICATE_API_TOKEN environment variables."
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
            
        # Try with Replicate first if API key is available
        if os.getenv('REPLICATE_API_TOKEN'):
            print("\n--- Trying Replicate API ---")
            try:
                result = await generate_image_with_replicate(
                    prompt=request.prompt,
                    child_name=request.child_name,
                    face_image_b64=request.face_image
                )
                
                if result and "image" in result:
                    print("✓ Successfully generated image with Replicate")
                    return {"image": result["image"], "source": "replicate"}
                else:
                    error_msg = result.get("error", "Unknown error from Replicate")
                    print(f"Replicate API error: {error_msg}")
                    
            except Exception as e:
                print(f"Replicate API call failed: {str(e)}")
                traceback.print_exc()
        else:
            print("Skipping Replicate (no API key found)")
        
        # Fall back to Stability AI if available
        if STABILITY_API_KEY:
            print("\n--- Trying Stability AI API ---")
            try:
                result = await _try_stability_ai(
                    prompt=request.prompt,
                    face_image_b64=request.face_image,
                    child_name=request.child_name
                )
                
                if isinstance(result, dict) and "image" in result:
                    print("✓ Successfully generated image with Stability AI")
                    return {"image": result["image"], "source": "stability-ai"}
                else:
                    error_msg = result.get("error", "Unknown error from Stability AI") if isinstance(result, dict) else "Invalid response from Stability AI"
                    print(f"Stability AI API error: {error_msg}")
                    
            except Exception as e:
                print(f"Stability AI API call failed: {str(e)}")
                traceback.print_exc()
        else:
            print("Skipping Stability AI (no API key found)")
        
        # If we get here, all attempts failed
        error_msg = "Failed to generate image with all available services. Check the logs for more details."
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/generate-story")
@app.websocket("/ws/{story_id}")
async def websocket_endpoint(websocket: WebSocket, story_id: str):
    await manager.connect(story_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(story_id, websocket)

@app.post("/generate-story")
async def generate_story(
    background_tasks: BackgroundTasks,
    child_name: str = Form(...),
    theme: str = Form(...),
    face_image: str = Form(None)
):
    """
    Generate an 8-page pirate story with Flapjack-style images.
    
    This endpoint generates the story and waits for all images to be ready before returning.
    """
    print("\n=== Starting story generation ===")
    print(f"Child name: {child_name}")
    print(f"Theme: {theme}")
    
    # Validate inputs
    if not child_name or not theme:
        raise HTTPException(status_code=400, detail="Child name and theme are required")
    
    # Generate the story text
    print("Generating story text...")
    story_data = generate_pirate_story_with_groq(child_name, theme)
    
    if not story_data:
        print("Using fallback story")
        story_data = create_fallback_pirate_story(child_name)
    
    # Create story parts with initial state
    parts = []
    story_id = str(uuid.uuid4())
    
    # Create StoryResponse object
    story_response = StoryResponse(
        title=story_data["title"],
        parts=[]
    )
    # Generate story text first
    try:
        story_response = await generate_story_text(child_name, theme)
    except Exception as e:
        logger.error(f"Error generating story text: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate story text")
    
    story_id = str(uuid.uuid4())
    
    # Store initial story data
    story_data = {
        "story_id": story_id,
        "title": story_response.title,
        "parts": [dict(part) for part in story_response.parts],
        "status": "generating_images",
        "completed": False
    }
    
    STORY_STORE[story_id] = story_data
    
    # Start image generation in background
    background_tasks.add_task(
        generate_story_images,
        story_id=story_id,
        story_response=story_response,
        face_image=face_image
    )
    
    return {
        "story_id": story_id,
        "title": story_response.title,
        "parts": [part.dict() for part in story_response.parts],
        "completed": False,
        "websocket_url": f"ws://{os.getenv('HOST', 'localhost')}:{os.getenv('PORT', '8000')}/ws/{story_id}"
    }

async def generate_story_images(story_id: str, story_response, face_image: str = None):
    """Generate images for all story parts sequentially"""
    try:
        story_data = STORY_STORE.get(story_id)
        if not story_data:
            logger.error(f"Story {story_id} not found in store")
            return

        # Initialize parts in story data if not present
        if "parts" not in story_data:
            story_data["parts"] = []
        
        for i, part in enumerate(story_response.parts):
            # Update status
            part["image_status"] = "processing"
            story_data["parts"][i] = part
            await manager.broadcast(story_id, {
                "part_index": i,
                "status": "processing"
            })

            try:
                # Generate image
                image_result = await generate_image(
                    prompt=part["image_prompt"],
                    face_image_b64=face_image,
                    child_name=story_data.get("child_name", "pirate"),
                    width=1024,
                    height=1024
                )
                
                if "error" in image_result:
                    part["image_status"] = "failed"
                    part["error"] = image_result["error"]
                else:
                    part["image_url"] = image_result.get("image")
                    part["image_status"] = "completed"
                
                # Update the part in the story data
                story_data["parts"][i] = part
                
                # Notify via WebSocket
                await manager.broadcast(story_id, {
                    "part_index": i,
                    "status": part["image_status"],
                    "image_url": part.get("image_url"),
                    "error": part.get("error")
                })
                
            except Exception as e:
                logger.error(f"Error generating image for part {i}: {str(e)}")
                part["image_status"] = "failed"
                part["error"] = str(e)
                story_data["parts"][i] = part
                await manager.broadcast(story_id, {
                    "part_index": i,
                    "status": "failed",
                    "error": str(e)
                })
            
            # Small delay between image generations
            await asyncio.sleep(1)
        
        # Mark story as completed
        story_data["status"] = "completed"
        story_data["completed"] = True
        await manager.broadcast(story_id, {"status": "completed"})
        
    except Exception as e:
        logger.error(f"Error in generate_story_images: {str(e)}")
        if story_id in STORY_STORE:
            story_data = STORY_STORE[story_id]
            story_data["status"] = "failed"
            story_data["error"] = str(e)
            await manager.broadcast(story_id, {
                "status": "failed",
                "error": str(e)
            })

class ImageTransformRequest(BaseModel):
    image_base64: str
    prompt: str = "A colorful cartoon version of the input image"
    strength: float = 0.7  # How much to transform the image (0.0 to 1.0)

@app.post("/transform-image")
async def transform_image(request: ImageTransformRequest):
    """
    Transform an image using an image-to-image model.
    Takes a base64-encoded image and returns a transformed version.
    """
    try:
        # Decode the base64 image
        image_data = base64.b64decode(request.image_base64.split(",")[-1])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_path = temp_file.name
        
        try:
            # Call the image generation function with the image path
            result = await generate_image_with_replicate(
                prompt=request.prompt,
                child_name="transformed",
                face_image_b64=request.image_base64,
                strength=request.strength
            )
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
                
            return {"image": result["image"]}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "groq_configured": bool(GROQ_API_KEY),
        "stability_configured": bool(STABILITY_API_KEY),
        "message": "Pirate Story Generator API is running"
    }

@app.get("/test/{name}/{theme}")
async def test_story(name: str, theme: str):
    """Quick test endpoint."""
    request = StoryRequest(child_name=name, theme=theme)
    return await generate_story(story_request=request)



async def generate_image_for_page(
    story_id: str, 
    page_index: int, 
    prompt: str, 
    child_name: str, 
    face_image: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    strength: float = 0.7
):
    """
    Generate an image for a story page and update the story state.
    
    Args:
        story_id: ID of the story to update
        page_index: Index of the page to generate an image for
        prompt: Text prompt for image generation
        child_name: Name of the child (for prompt enhancement)
        face_image: Optional base64-encoded image for image-to-image
        width: Width of the output image (default: 1024)
        height: Height of the output image (default: 1024)
        strength: How much to transform the input image (0.0 to 1.0, default: 0.7)
    """
    if story_id not in story_states:
        print(f"Story ID {story_id} not found in story_states")
        return {
            "story_id": story_id,
            "page_index": page_index,
            "status": "failed",
            "error": "Story not found"
        }
    
    story_state = story_states[story_id]
    
    try:
        # Update status to 'generating'
        if page_index < len(story_state["story"].parts):
            story_state["story"].parts[page_index].image_status = "generating"
            story_state["updated_at"] = time.time()
            await broadcast_update(story_id)
        
        # Generate the image
        image_result = await generate_image(
            prompt=prompt,
            child_name=child_name,
            face_image_b64=face_image,
            width=width,
            height=height,
            strength=strength
        )
        
        # Update the story with the result
        if "image" in image_result:
            if page_index < len(story_state["story"].parts):
                story_state["story"].parts[page_index].image_url = image_result["image"]
                story_state["story"].parts[page_index].image_status = "completed"
                story_state["completed_pages"] = story_state.get("completed_pages", 0) + 1
                story_state["updated_at"] = time.time()
                print(f"Generated image for page {page_index}")
                
                # Broadcast the update
                await broadcast_update(story_id)
                
                return {
                    "story_id": story_id,
                    "page_index": page_index,
                    "status": "completed",
                    "source": image_result.get("source", "unknown")
                }
        
        # If we get here, image generation failed
        error_msg = image_result.get("error", "Unknown error")
        print(f"Failed to generate image for page {page_index}: {error_msg}")
        
        if page_index < len(story_state["story"].parts):
            story_state["story"].parts[page_index].error = error_msg
            story_state["story"].parts[page_index].image_status = "failed"
            story_state["completed_pages"] = story_state.get("completed_pages", 0) + 1
            story_state["updated_at"] = time.time()
            
            # Broadcast the update
            await broadcast_update(story_id)
        
        return {
            "story_id": story_id,
            "page_index": page_index,
            "status": "failed",
            "error": error_msg
        }
        
    except Exception as e:
        error_msg = f"Error generating image for page {page_index}: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        
        if page_index < len(story_state["story"].parts):
            story_state["story"].parts[page_index].error = error_msg
            story_state["story"].parts[page_index].image_status = "failed"
            story_state["completed_pages"] = story_state.get("completed_pages", 0) + 1
            story_state["updated_at"] = time.time()
            
            # Broadcast the update
            await broadcast_update(story_id)
        
        return {
            "story_id": story_id,
            "page_index": page_index,
            "status": "failed",
            "error": error_msg
        }

async def broadcast_update(story_id: str):
    """
    Send updates to all connected WebSocket clients for this story.
    
    Args:
        story_id: The ID of the story to broadcast updates for
    """
    if story_id not in active_connections:
        return
        
    story_data = story_states.get(story_id)
    if not story_data or "story" not in story_data:
        return
    
    # Prepare the update data
    update_data = {
        "story_id": story_id,
        "title": story_data["title"],
        "parts": [part.dict() for part in story_data["story"].parts],
        "completed": story_data.get("completed", False),
        "completed_pages": story_data.get("completed_pages", 0),
        "total_pages": story_data.get("total_pages", 0)
    }
    
    # Send the update to all connected clients
    disconnected_clients = []
    
    for websocket in active_connections[story_id]:
        try:
            await websocket.send_json({
                "type": "update",
                "data": update_data
            })
        except Exception as e:
            print(f"Error broadcasting to WebSocket: {str(e)}")
            disconnected_clients.append(websocket)
    
    # Clean up disconnected clients
    if disconnected_clients:
        for websocket in disconnected_clients:
            if websocket in active_connections[story_id]:
                active_connections[story_id].remove(websocket)
        
        if not active_connections[story_id]:
            del active_connections[story_id]

@app.websocket("/ws/{story_id}")
async def websocket_endpoint(websocket: WebSocket, story_id: str):
    """
    WebSocket endpoint for real-time updates on story generation.
    
    Args:
        websocket: The WebSocket connection
        story_id: The ID of the story to get updates for
    """
    await websocket.accept()
    
    # Add client to active connections
    if story_id not in active_connections:
        active_connections[story_id] = set()
    active_connections[story_id].add(websocket)
    
    try:
        # Send current state immediately if available
        if story_id in story_states:
            story_data = story_states[story_id]
            update_data = {
                "story_id": story_id,
                "title": story_data["title"],
                "parts": [part.dict() for part in story_data["story"].parts],
                "completed": story_data.get("completed", False),
                "completed_pages": story_data.get("completed_pages", 0),
                "total_pages": story_data.get("total_pages", 0)
            }
            
            await websocket.send_json({
                "type": "update",
                "data": update_data
            })
        else:
            # If story doesn't exist, send an error and close the connection
            await websocket.send_json({
                "type": "error",
                "error": "Story not found"
            })
            await websocket.close()
            return
            
        # Keep connection open and handle incoming messages
        while True:
            try:
                # Set a timeout to periodically check if the connection is still alive
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle ping/pong for keepalive
                if data == "ping":
                    await websocket.send_text("pong")
                    
            except asyncio.TimeoutError:
                # Send a ping to check if the connection is still alive
                try:
                    await websocket.send_text("ping")
                except:
                    # Connection is dead, break the loop
                    break
                    
            except WebSocketDisconnect:
                # Client disconnected normally
                break
                
            except Exception as e:
                print(f"WebSocket receive error: {str(e)}")
                break
                
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        
    finally:
        # Clean up on disconnect
        if story_id in active_connections and websocket in active_connections[story_id]:
            active_connections[story_id].remove(websocket)
            if not active_connections[story_id]:
                del active_connections[story_id]
        
        try:
            await websocket.close()
        except:
            pass

@app.get("/story/{story_id}")
async def get_story(story_id: str):
    """
    Get the current state of a story.
    
    Args:
        story_id: The ID of the story to retrieve
        
    Returns:
        The story data including title, parts, and generation status
    """
    story_data = story_states.get(story_id)
    if not story_data or "story" not in story_data:
        raise HTTPException(status_code=404, detail="Story not found")
    
    return {
        "story_id": story_id,
        "title": story_data["title"],
        "parts": [part.dict() for part in story_data["story"].parts],
        "completed": story_data.get("completed", False),
        "completed_pages": story_data.get("completed_pages", 0),
        "total_pages": story_data.get("total_pages", 0),
        "updated_at": story_data.get("updated_at", 0)
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable or default to 8000 for local development
    port = int(os.environ.get('PORT', 8000))
    
    print("🏴‍☠️ Starting Pirate Story Generator API...")
    print(f"Groq API configured: {bool(GROQ_API_KEY)}")
    print(f"Stability AI configured: {bool(STABILITY_API_KEY)}")
    print(f"Server starting on port {port}")
    print("Ready to generate pirate adventures!")
    
    # Note: For production, Render will use the start.sh script which runs:
    # uvicorn app.main:app --host 0.0.0.0 --port $PORT
    uvicorn.run(app, host="0.0.0.0", port=port)