from typing import List, Optional, Dict, Any, Union
import base64
import io
import os
import traceback
import uuid
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import requests
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
import mediapipe as mp
from io import BytesIO
import asyncio
import replicate
import time

# Uncomment these if you want to use Hugging Face in the future
# from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
# import torch
# from transformers import pipeline as hf_pipeline

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get allowed origins from environment variable or use default
FRONTEND_URL = os.getenv('FRONTEND_URL', 'https://mystoria-alpha.vercel.app')

app = FastAPI(title="Pirate Story Generator API")

# CORS middleware configuration
# In development, allow all origins for easier debugging
# In production, you should restrict this to your frontend domain
is_production = os.getenv('ENVIRONMENT', 'development') == 'production'

# Default allowed origins (for production)
allowed_origins = [
    "https://mystoria-alpha.vercel.app",
    "https://mystoria-alpha.vercel.app/",
]

# Development origins (added if not in production)
if not is_production:
    development_origins = [
        "http://localhost:*",
        "http://127.0.0.1:*",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://localhost:5173",
        "https://localhost:5174",
        "https://127.0.0.1:5173",
        "https://127.0.0.1:5174",
    ]
    allowed_origins.extend(development_origins)

# Add any additional origins from environment variable
if FRONTEND_URL:
    # Add both with and without trailing slash
    allowed_origins.extend([FRONTEND_URL.rstrip('/'), f"{FRONTEND_URL.rstrip('/')}/"])

# Remove duplicates
allowed_origins = list(set(allowed_origins))

# Log allowed origins for debugging
print("\n" + "="*50)
print(f"Environment: {'PRODUCTION' if is_production else 'DEVELOPMENT'}")
print(f"Frontend URL: {FRONTEND_URL}")
print(f"Allowed Origins: {json.dumps(allowed_origins, indent=2)}")
print("="*50 + "\n")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex='https?://.*\.?vercel\.app/?',  # Allow any Vercel deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

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

class StoryRequest(BaseModel):
    child_name: str = ""
    theme: str = ""
    face_image: Optional[str] = None  # Base64 encoded face image

class PirateImageRequest(BaseModel):
    """Request model for pirate image generation"""
    image_base64: str
    prompt: str = "Turn this into a pirate-themed image"
    child_name: str = "pirate"

class StoryResponse(BaseModel):
    title: str
    parts: List[StoryPart]

def generate_pirate_story_with_groq(child_name: str, theme: str):
    """Generate an 8-page pirate story using Groq API."""
    print(f"\n=== Generating pirate story for Captain {child_name} ===")
    
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
            print(f"API Error: {result['error']}")
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

async def generate_image_with_replicate(prompt: str, face_image_b64: Optional[str], child_name: str) -> Optional[str]:
    """Generate image using Replicate with face consistency."""
    print("\n=== Starting Replicate image generation ===")
    print(f"Prompt: {prompt}")
    
    if not REPLICATE_API_KEY:
        error_msg = "No Replicate API key found"
        print(error_msg)
        return {"error": error_msg}
    
    if not face_image_b64:
        error_msg = "No face image provided"
        print(error_msg)
        return {"error": error_msg}
    
    try:
        # Create the enhanced prompt with Flapjack style
        enhanced_prompt = create_flapjack_style_prompt(prompt, child_name)
        print(f"Enhanced prompt: {enhanced_prompt}")
        
        # Fix base64 padding and remove data URL prefix
        try:
            face_image_b64 = fix_base64_padding(face_image_b64)
            # Decode the image
            image_data = base64.b64decode(face_image_b64)
            print(f"Successfully decoded image data: {len(image_data)} bytes")
            
            # Save to a temporary file
            temp_img_path = "/tmp/face_input.jpg"
            with open(temp_img_path, "wb") as f:
                f.write(image_data)
                
        except Exception as e:
            error_msg = f"Error processing image data: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
        
        print("Sending request to Replicate...")
        
        # Run the Replicate model
        output = replicate.run(
            "lucataco/sdxl-controlnet:98e043dc665c8fca8dc37668cb847005c4d59ccb45e3d9f4fcb557dcdf670a63",
            input={
                "image": open(temp_img_path, "rb"),
                "prompt": enhanced_prompt,
                "num_inference_steps": 30,
                "controlnet_conditioning_scale": 0.8,
                "guidance_scale": 7.5,
                "negative_prompt": "blurry, low quality, distorted, disfigured, extra limbs, extra fingers, bad anatomy",
                "width": 1024,
                "height": 1024
            }
        )
        
        print(f"Replicate output: {output}")
        
        if not output or not isinstance(output, list) or not output[0]:
            error_msg = "No valid output from Replicate"
            print(error_msg)
            return {"error": error_msg}
        
        # Get the image URL from Replicate
        image_url = output[0]
        print(f"Successfully generated image at: {image_url}")
        
        # Download the image and convert to base64
        response = requests.get(image_url)
        if response.status_code == 200:
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            return f"data:image/png;base64,{image_base64}"
        else:
            error_msg = f"Failed to download image from Replicate: {response.status_code}"
            print(error_msg)
            return {"error": error_msg}
            
    except Exception as e:
        error_msg = f"Error with Replicate: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        return {"error": error_msg}

async def generate_image_with_face(prompt: str, face_image_b64: Optional[str], child_name: str) -> Optional[str]:
    """Generate image using available services with detailed error reporting."""
    print("\n=== Starting image generation ===")
    print(f"Prompt: {prompt}")
    
    # Track which services were attempted and why they failed
    attempts = []
    
    # First try Stability AI if key is available
    if STABILITY_API_KEY:
        print("\n--- Trying Stability AI ---")
        result = await _try_stability_ai(prompt, face_image_b64, child_name)
        if not isinstance(result, dict) or 'error' not in result:
            print("✓ Successfully generated with Stability AI")
            return result
            
        error_msg = f"Stability AI failed: {result.get('error')}"
        print(f"✗ {error_msg}")
        attempts.append({"service": "Stability AI", "error": result.get('error')})
        
        # If we have a detailed response, log it
        if 'response' in result:
            print(f"Stability AI response: {result['response']}")
    else:
        attempts.append({"service": "Stability AI", "error": "API key not configured"})
    
    # Fall back to Replicate if Stability AI fails or isn't configured
    if REPLICATE_API_KEY:
        print("\n--- Trying Replicate ---")
        result = await generate_image_with_replicate(prompt, face_image_b64, child_name)
        if not isinstance(result, dict) or 'error' not in result:
            print("✓ Successfully generated with Replicate")
            return result
            
        error_msg = f"Replicate failed: {result.get('error')}"
        print(f"✗ {error_msg}")
        attempts.append({"service": "Replicate", "error": result.get('error')})
    else:
        attempts.append({"service": "Replicate", "error": "API key not configured"})
    
    # Generate detailed error message
    error_details = "\n".join([
        f"- {attempt['service']}: {attempt['error']}" 
        for attempt in attempts
    ])
    
    if not attempts:
        final_error = "No image generation services were attempted. Check your API keys."
    else:
        final_error = f"All image generation attempts failed. Here's what happened:\n{error_details}"
    
    print(f"\n=== Image Generation Failed ===\n{final_error}")
    return {"error": final_error, "attempts": attempts}

async def _try_stability_ai(prompt: str, face_image_b64: Optional[str], child_name: str) -> Union[str, Dict[str, str]]:
    """Try to generate an image using Stability AI."""
    if not face_image_b64:
        return {"error": "No face image provided"}
    
    try:
        # Create the enhanced prompt with Flapjack style
        enhanced_prompt = create_flapjack_style_prompt(prompt, child_name)
        print(f"Enhanced prompt: {enhanced_prompt[:100]}...")  # Print first 100 chars
        
        # Fix base64 padding and remove data URL prefix
        try:
            face_image_b64 = fix_base64_padding(face_image_b64)
            # Decode and validate the image
            image_data = base64.b64decode(face_image_b64, validate=True)
            print(f"Successfully decoded image data: {len(image_data)} bytes")
        except Exception as e:
            error_msg = f"Error processing image data: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
        
        # Prepare the request data
        files = {"init_image": ("face.jpg", image_data, "image/jpeg")}
        data = {
            "image_strength": 0.4,
            "init_image_mode": "IMAGE_STRENGTH",
            "text_prompts[0][text]": enhanced_prompt,
            "text_prompts[0][weight]": 1.0,
            "cfg_scale": 8,
            "samples": 1,
            "steps": 35,
            "style_preset": "cartoon"
        }
        
        print("Sending request to Stability AI...")
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {STABILITY_API_KEY}"
            },
            files=files,
            data=data,
            timeout=60  # 60 seconds timeout
        )
        
        print(f"Status code: {response.status_code}")
        
        try:
            result = response.json()
            print(f"Response keys: {list(result.keys())}")
            
            if not response.ok:
                error_msg = f"Stability AI API error: {result.get('name', 'Unknown error')} - {result.get('message', 'No error message')}"
                print(error_msg)
                return {"error": error_msg}
            
            if result.get("artifacts"):
                print(f"Successfully generated {len(result['artifacts'])} artifacts")
                if result['artifacts'][0].get('base64'):
                    return f"data:image/png;base64,{result['artifacts'][0]['base64']}"
                else:
                    error_msg = "No base64 data in artifacts"
            else:
                error_msg = "No artifacts in response"
                
            print(error_msg)
            return {"error": error_msg}
            
        except Exception as e:
            error_msg = f"Error parsing response: {str(e)}"
            print(f"{error_msg}\nResponse content: {response.text[:500]}...")
            return {"error": error_msg, "response": response.text[:500]}
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {str(e)}"
        print(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
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

@app.post("/generate-story", response_model=StoryResponse)
async def generate_story(
    request: Request,
    child_name: str = Form(None),
    theme: str = Form(None),
    face_image: str = Form(None)
):
    """Generate an 8-page pirate story with face-consistent Flapjack-style images."""
    try:
        print("\n=== Starting story generation ===")
        
        # Try to get JSON data if form data is not provided
        content_type = request.headers.get('content-type', '')
        if not child_name or not theme and 'application/json' in content_type:
            try:
                json_data = await request.json()
                child_name = json_data.get('child_name') or child_name
                theme = json_data.get('theme') or theme
                face_image = json_data.get('face_image', face_image)
            except Exception as e:
                print(f"Error parsing JSON data: {str(e)}")
        
        print(f"Child name: {child_name}")
        print(f"Theme: {theme}")
        print(f"Face image provided: {bool(face_image)}")
        
        if not child_name or not theme:
            error_msg = "Child name and theme are required"
            print(f"Validation error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Generate story using Groq
        print("\nGenerating story with Groq...")
        try:
            story = generate_pirate_story_with_groq(child_name, theme)
            if not story or "pages" not in story:
                print("No valid story data returned from Groq, using fallback story")
                story = create_fallback_pirate_story(child_name)
            print(f"Successfully generated story with {len(story.get('pages', []))} pages")
        except Exception as e:
            print(f"Error generating story with Groq: {str(e)}")
            print(traceback.format_exc())
            story = create_fallback_pirate_story(child_name)
        
        # Process each page to generate images
        print("\nProcessing pages and generating images...")
        
        if face_image:
            print("Face image detected, generating images with face consistency...")
            for i, page in enumerate(story.get('pages', [])):
                try:
                    print(f"\n--- Processing page {i+1}/{len(story['pages'])} ---")
                    print(f"Prompt: {page.get('image_prompt', 'No prompt')}")
                    
                    image_result = await generate_image_with_face(
                        page.get("image_prompt", ""),
                        face_image,
                        child_name
                    )
                    
                    if isinstance(image_result, dict) and 'error' in image_result:
                        print(f"Error generating image: {image_result['error']}")
                        if 'response' in image_result:
                            print(f"Response: {image_result['response']}")
                        # Set a placeholder image URL instead of None
                        page['image_url'] = "https://via.placeholder.com/1024x1024?text=Image+Generation+Failed"
                    else:
                        page['image_url'] = image_result
                        print(f"Successfully generated image for page {i+1}")
                    
                    # Add a small delay between image generations
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    print(f"Error processing page {i+1}: {str(e)}")
                    print(traceback.format_exc())
                    # Set a placeholder image URL instead of None
                    page['image_url'] = "https://via.placeholder.com/1024x1024?text=Image+Generation+Error"
        
        # Transform the story to match the frontend's expected format
        response = {
            "title": story.get("title", f"{child_name}'s Pirate Adventure"),
            "parts": [
                {
                    "text": page.get("story_text", ""),
                    "image_url": page.get("image_url"),
                    "image_prompt": page.get("image_prompt", "")
                }
                for page in story.get("pages", [])
            ]
        }
        
        print("\n=== Story generation completed successfully ===")
        return response
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        error_msg = f"Unexpected error in generate_story: {str(e)}"
        print(f"\n!!! {error_msg}")
        print(traceback.format_exc())
        # Return a fallback response instead of 500 to keep the frontend working
        fallback_story = create_fallback_pirate_story(child_name or "Pirate")
        return {
            "title": fallback_story["title"],
            "parts": [
                {
                    "text": page["story_text"],
                    "image_url": None,
                    "image_prompt": page.get("image_prompt", "")
                }
                for page in fallback_story["pages"]
            ]
        }

@app.post("/upload-face")
async def upload_face(file: UploadFile = File(...)):
    """Upload and convert face image to base64."""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Convert RGBA to RGB if needed
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        if image.width > 512 or image.height > 512:
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=90)
        face_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {"face_image": f"data:image/jpeg;base64,{face_b64}"}
        
    except Exception as e:
        print(f"Error in upload_face: {str(e)}")
        print(traceback.format_exc())
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

@app.post("/api/generate-pirate-image", response_model=Dict[str, str])
async def generate_pirate_image(request: PirateImageRequest):
    """
    Generate a pirate-themed version of the input image.
    
    Parameters:
    - image_base64: Base64 encoded image data
    - prompt: Description of how to transform the image (default: pirate theme)
    - child_name: Name to use for character consistency (default: "pirate")
    
    Returns:
    - Dictionary with the generated image in base64 format
    """
    print(f"Starting image generation with prompt: {request.prompt}")
    
    # Validate base64 data
    if not request.image_base64 or not request.image_base64.strip():
        raise HTTPException(status_code=400, detail="No image data provided")
        
    try:
        # Try to decode the base64 to verify it's valid
        try:
            # Remove data URL prefix if present
            if "," in request.image_base64:
                request.image_base64 = request.image_base64.split(",", 1)[1]
            base64.b64decode(request.image_base64, validate=True)
        except Exception as e:
            print(f"Invalid base64 data: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
            
        print("Base64 data validated, starting image generation...")
        
        # Generate the image using the existing function
        result = await generate_image_with_face(
            prompt=request.prompt,
            face_image_b64=request.image_base64,
            child_name=request.child_name
        )
        
        print(f"Image generation result: {'Success' if result and 'image' in result else 'Failed'}")
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="No result from image generation service"
            )
            
        if "error" in result:
            error_detail = result.get("error", "Unknown error during image generation")
            print(f"Error in image generation: {error_detail}")
            raise HTTPException(
                status_code=500,
                detail=error_detail
            )
            
        if not result.get("image"):
            raise HTTPException(
                status_code=500,
                detail="Image generation succeeded but no image data was returned"
            )
            
        print("Image generated successfully")
        return {"image_base64": result.get("image")}
        
    except HTTPException:
        # Re-raise HTTP exceptions as they are
        raise
        
    except Exception as e:
        error_msg = f"Error in generate_pirate_image: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

if __name__ == "__main__":
    import uvicorn
    print("🏴‍☠️ Starting Pirate Story Generator API...")
    print(f"Groq API configured: {bool(GROQ_API_KEY)}")
    print(f"Stability AI configured: {bool(STABILITY_API_KEY)}")
    print("Ready to generate pirate adventures!")
    uvicorn.run(app, host="0.0.0.0", port=8000)