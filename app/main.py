from typing import List, Optional, Dict, Any, Union
import base64
import io
import os
import logging
import time
import tempfile
import asyncio
import json
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
import replicate
import requests
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for story states
STORY_STORE: Dict[str, dict] = {}
active_connections: Dict[str, List[WebSocket]] = {}

# Story generation models
class StoryPart(BaseModel):
    text: str
    image_prompt: str
    image_url: Optional[str] = None
    image_status: str = "pending"  # pending, processing, completed, failed
    error: Optional[str] = None

class StoryRequest(BaseModel):
    child_name: str
    theme: str = "pirate"
    face_image: Optional[str] = None

class StoryResponse(BaseModel):
    story_id: str
    title: str
    parts: List[StoryPart]
    status: str = "text_generated"  # text_generated, generating_images, completed, failed
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    child_name: str
    theme: str
    face_image: Optional[str] = None

# In-memory story store
STORY_STORE: Dict[str, StoryResponse] = {}

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# API Configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')
STABILITY_API_HOST = 'https://api.stability.ai'
REPLICATE_API_KEY = os.getenv('REPLICATE_API_KEY')

# Configure Replicate
if REPLICATE_API_KEY:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY

# Frontend URL configuration
FRONTEND_URL = os.getenv('FRONTEND_URL', 'https://mystoria-alpha.vercel.app')
is_production = os.getenv('ENVIRONMENT', 'development') == 'production'

# Initialize FastAPI app
app = FastAPI(title="Kids Story Generator API")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# CORS Configuration
allowed_origins = []

# Add production frontend URL
if FRONTEND_URL:
    allowed_origins.extend([
        FRONTEND_URL.rstrip('/'),
        f"{FRONTEND_URL.rstrip('/')}/"
    ])

# Add development origins if not in production
if not is_production:
    development_origins = [
        "http://localhost:3000", "http://localhost:8000", "http://localhost:5173", "http://localhost:5174",
        "http://127.0.0.1:3000", "http://127.0.0.1:8000", "http://127.0.0.1:5173", "http://127.0.0.1:5174",
        "https://localhost:3000", "https://localhost:8000", "https://localhost:5173", "https://localhost:5174"
    ]
    allowed_origins.extend(development_origins)

# Add Vercel preview URLs
allowed_origins.append(r'https://.*\.vercel\.app')

# Remove duplicates
allowed_origins = list(set(allowed_origins))

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Pydantic Models
class StoryPart(BaseModel):
    page_number: int
    text: str
    image_prompt: str
    image_url: Optional[str] = None
    image_status: str = "pending"  # pending, processing, completed, failed
    error: Optional[str] = None

class StoryResponse(BaseModel):
    story_id: str
    title: str
    parts: List[StoryPart]
    status: str = "text_generated"  # text_generated, generating_images, completed, failed
    created_at: float
    updated_at: float

class GenerateStoryRequest(BaseModel):
    child_name: str
    theme: str = "pirate"

class PirateImageRequest(BaseModel):
    face_image: str  # Base64 encoded image
    prompt: str
    child_name: str

class GenerateImageRequest(BaseModel):
    story_id: str
    page_number: int
    user_face_image: str  # Base64 encoded image
    prompt: str

class BatchGenerateImagesRequest(BaseModel):
    story_id: str
    user_face_image: str  # Base64 encoded image

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
            if websocket in self.active_connections[story_id]:
                self.active_connections[story_id].remove(websocket)
            if not self.active_connections[story_id]:
                del self.active_connections[story_id]

    async def broadcast(self, story_id: str, message: dict):
        if story_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[story_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending WebSocket message: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                if conn in self.active_connections[story_id]:
                    self.active_connections[story_id].remove(conn)

manager = ConnectionManager()

# ================================
# STORY GENERATION ENDPOINTS
# ================================

@app.post("/generate-story")
async def generate_story(
    request: StoryRequest,
    background_tasks: BackgroundTasks
):
    """Generate story text with Groq API"""
    story_id = str(uuid.uuid4())
    
    try:
        # Generate story text using Groq
        story_data = generate_story_with_groq(
            child_name=request.child_name,
            theme=request.theme
        )
        
        if not story_data:
            logger.error("Failed to generate story with Groq, using fallback story")
            story_data = create_fallback_story(
                child_name=request.child_name,
                theme=request.theme
            )
            
        logger.info(f"Generated story with {len(story_data.get('pages', []))} pages")
        
        # Create story object
        story = StoryResponse(
            story_id=story_id,
            title=story_data["title"],
            child_name=request.child_name,
            theme=request.theme,
            face_image=request.face_image,
            parts=[
                StoryPart(
                    text=part["text"],
                    image_prompt=part["image_prompt"]
                ) for part in story_data["parts"]
            ]
        )
        
        # Store story
        STORY_STORE[story_id] = story
        
        # Start generating images in background
        background_tasks.add_task(
            generate_story_images,
            story_id=story_id
        )
        
        # Return basic story info without image URLs
        return {
            "story_id": story_id,
            "title": story.title,
            "status": story.status,
            "parts": [{"text": p.text, "image_prompt": p.image_prompt} for p in story.parts]
        }
        
    except Exception as e:
        logger.error(f"Error generating story: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate story")

async def generate_story_images(story_id: str):
    """Generate images for all parts of a story"""
    story = STORY_STORE.get(story_id)
    if not story:
        return
    
    try:
        story.status = "generating_images"
        story.updated_at = time.time()
        
        for i, part in enumerate(story.parts):
            try:
                # Update status
                part.image_status = "processing"
                story.updated_at = time.time()
                
                # Generate image
                result = await generate_single_image(
                    prompt=part.image_prompt,
                    child_name=story.child_name,
                    face_image_b64=story.face_image,
                    theme=story.theme
                )
                
                if "image" in result:
                    part.image_url = result["image"]
                    part.image_status = "completed"
                else:
                    part.image_status = "failed"
                    part.error = result.get("error", "Unknown error")
                
            except Exception as e:
                logger.error(f"Error generating image for part {i}: {str(e)}")
                part.image_status = "failed"
                part.error = str(e)
            
            story.updated_at = time.time()
            
        # Update final status
        if all(p.image_status == "completed" for p in story.parts):
            story.status = "completed"
        else:
            story.status = "completed_with_errors"
            
    except Exception as e:
        logger.error(f"Error in generate_story_images: {str(e)}")
        story.status = "failed"
    finally:
        story.updated_at = time.time()

@app.get("/story/{story_id}")
async def get_story(story_id: str):
    """Get the current status of a story"""
    story = STORY_STORE.get(story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    return story

# ================================
# STORY TEXT GENERATION FUNCTIONS
# ================================

def generate_story_with_groq(child_name: str, theme: str = "pirate") -> Optional[Dict]:
    """
    Generate an 8-page story using Groq API.
    
    Args:
        child_name: Name of the child protagonist
        theme: Theme of the story (e.g., "pirate", "space", "fairy")
    
    Returns:
        Dictionary containing story data or None if failed
    """
    prompt = f"""Create an 8-page children's {theme} adventure story about {child_name}. 

IMPORTANT: {child_name} is a young {theme} child (8-12 years old) who will be the main character in every page.

Each page should be a complete mini-adventure that connects to the overall story.

Format your response as valid JSON like this:
{{
    "title": "Captain {child_name}'s [Adventure Name]",
    "pages": [
        {{
            "page_number": 1,
            "story_text": "A full paragraph (4-5 sentences) describing this part of the adventure...",
            "image_prompt": "Detailed visual description for this page featuring young {theme} {child_name} - describe the scene, {child_name}'s actions, environment, other characters, and objects. Focus on visual elements that can be illustrated."
        }}
    ]
}}

Story requirements:
- Each page should have 4-5 sentences of story text
- Each page should work as a mini-adventure but connect to the overall story
- Focus on positive themes like: adventure, friendship, helping others, solving puzzles, discovery
- Keep it age-appropriate for children 6-12
- Make {child_name} the brave hero of each page
- Ensure the story has a clear beginning, middle, and satisfying end

For image_prompt, describe the scene visually without mentioning art style. Focus on:
- {child_name}'s appearance and actions in {theme} costume/outfit
- The environment and setting
- Other characters present
- Important objects or elements in the scene
- The mood and atmosphere"""

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
        logger.info(f"Generating story for {child_name} with theme: {theme}")
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        result = response.json()
        
        if 'error' in result:
            logger.error(f"Groq API error: {result['error']}")
            return None
            
        if 'choices' not in result or not result['choices']:
            logger.error("No choices in Groq response")
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
                logger.error("Could not extract JSON from Groq response")
                return None
                    
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return None

def create_fallback_story(child_name: str, theme: str = "pirate") -> Dict:
    """Create a fallback story when API fails."""
    return {
        "title": f"Captain {child_name}'s Treasure Island Adventure",
        "pages": [
            {
                "page_number": 1,
                "story_text": f"Young {child_name} discovered an ancient treasure map hidden in their grandmother's attic. The map was drawn on yellowed parchment and showed a mysterious island with an X marking buried treasure. {child_name} decided this was the perfect day for a grand pirate adventure. They quickly put on their favorite pirate hat and grabbed their wooden sword.",
                "image_prompt": f"Young pirate {child_name} in dusty attic holding old treasure map, wearing pirate hat and vest, excited expression, surrounded by old trunks and spider webs, warm sunlight streaming through window"
            },
            {
                "page_number": 2,
                "story_text": f"Captain {child_name} built a magnificent pirate ship from cardboard boxes and set sail across the backyard ocean. The ship had a tall mast with a black flag featuring a friendly skull. As they sailed, {child_name} sang pirate songs and watched for sea monsters. The wind filled their imagination with dreams of adventure and treasure.",
                "image_prompt": f"Pirate captain {child_name} standing proudly on cardboard ship deck in sunny backyard, pirate flag flying above, wooden sword at side, looking out at imaginary sparkling blue ocean"
            },
            {
                "page_number": 3,
                "story_text": f"The mysterious treasure island appeared on the horizon, covered with tall palm trees and golden beaches. {child_name} carefully navigated through dangerous waters filled with friendly rubber ducky sea creatures. As they approached the shore, colorful parrots welcomed them with cheerful squawks. The island looked exactly like the one on the treasure map.",
                "image_prompt": f"Pirate {child_name} approaching tropical island on ship, lush palm trees and golden beach visible, colorful parrots flying around, cute rubber ducky creatures swimming in crystal blue water"
            },
            {
                "page_number": 4,
                "story_text": f"Following the treasure map through the jungle, {child_name} discovered a hidden cave behind a waterfall. The cave entrance sparkled with colorful crystals and glowed with mysterious light. {child_name} bravely entered the cave, using their flashlight to guide the way. Strange but friendly cave creatures watched from the shadows, curious about this young pirate explorer.",
                "image_prompt": f"Brave pirate {child_name} entering magical crystal cave behind waterfall, flashlight in hand, rainbow crystals sparkling on cave walls, friendly glowing creatures peeking from shadows with curious eyes"
            },
            {
                "page_number": 5,
                "story_text": f"Deep inside the cave, {child_name} met Captain Silverbeard, a friendly ghost pirate who had been guarding the treasure for a hundred years. Captain Silverbeard told amazing stories of sea adventures and pirate legends. He was impressed by {child_name}'s courage and kindness. The ghost captain decided that {child_name} was worthy of learning the treasure's secret.",
                "image_prompt": f"Young pirate {child_name} talking with friendly translucent ghost pirate Captain Silverbeard in glowing cave, both wearing pirate outfits, ancient treasure chests visible in background, warm golden light surrounding them"
            },
            {
                "page_number": 6,
                "story_text": f"Captain Silverbeard revealed that the real treasure wasn't gold or jewels, but a magical compass that always points toward adventure and friendship. The compass glowed with warm light and hummed with ancient magic. {child_name} learned that the greatest treasures are the friends you make and the adventures you share. The ghost captain smiled proudly as he passed on this wisdom.",
                "image_prompt": f"Ghost pirate Captain Silverbeard giving glowing magical compass to young pirate {child_name}, warm golden light radiating from compass, both characters smiling, ancient treasure cave setting with mystical atmosphere"
            },
            {
                "page_number": 7,
                "story_text": f"Using the magical compass, {child_name} helped rescue a family of lost sea turtles who couldn't find their way home. The compass led them through coral reefs and past sleeping sea dragons to the turtles' underwater city. The grateful turtle family invited {child_name} to visit anytime. {child_name} realized that helping others felt better than finding any treasure.",
                "image_prompt": f"Pirate {child_name} swimming underwater with sea turtle family, magical compass glowing in hand, colorful coral reefs and peaceful sleeping sea dragons in background, underwater city with bubble domes visible"
            },
            {
                "page_number": 8,
                "story_text": f"Captain {child_name} sailed home as the sun set over the ocean, painting the sky in brilliant oranges and purples. Back in their backyard, they carefully put away their pirate hat and treasure map, but kept the magical compass close. {child_name} fell asleep that night dreaming of tomorrow's adventures, knowing that the greatest treasure is a heart full of courage and kindness.",
                "image_prompt": f"Pirate {child_name} back home at beautiful sunset, holding magical compass close to heart, cardboard pirate ship in background, warm orange and purple sky, peaceful and happy expression showing contentment"
            }
        ]
    }

# ================================
# IMAGE GENERATION FUNCTIONS
# ================================

def create_enhanced_image_prompt(base_prompt: str, child_name: str, theme: str = "pirate") -> str:
    """
    Enhance the image prompt with consistent styling and character appearance.
    
    Args:
        base_prompt: The base prompt describing the scene
        child_name: Name of the child character
        theme: Theme of the story (e.g., "pirate")
    
    Returns:
        Enhanced prompt with style and character consistency instructions
    """
    style_prompt = f"""
    {base_prompt}

    Character consistency: {child_name} always appears as the same young {theme} child (8-12 years old) 
    with consistent facial features throughout all images. {child_name} wears {theme}-themed outfit 
    (bandana, vest, boots for pirate theme) and has an adventurous, cheerful expression.

    Art style: Children's book illustration style, colorful and whimsical, 2D cartoon look similar to 
    animated movies. Warm, friendly, and inviting visual style suitable for kids.

    Visual aesthetics: Bright, vibrant colors with good contrast. Child-friendly and non-scary imagery. 
    The scene should feel magical and adventurous while being appropriate for young children.

    Quality: High quality, detailed illustration, sharp focus, well-composed scene with clear visual hierarchy.
    
    Character appearance: The same child's face should be recognizable in every image, just in different 
    {theme} costumes and scenarios.
    """
    
    return style_prompt.strip()

async def generate_image_with_stability_ai(
    prompt: str, 
    child_name: str, 
    face_image_b64: str,
    theme: str = "pirate",
    strength: float = 0.7,
    width: int = 1024,
    height: int = 1024
) -> Dict[str, Any]:
    """
    Generate image using Stability AI with image-to-image transformation.
    
    Args:
        prompt: Text prompt for image generation
        child_name: Name of the child character
        face_image_b64: Base64-encoded face image
        theme: Theme of the story
        strength: Transformation strength (0.0 to 1.0)
        width: Output image width
        height: Output image height
    Returns:
        Dictionary with 'image' (base64) or 'error' key
    """
    if not STABILITY_API_KEY:
        return {"error": "Stability API key not configured"}
    
    enhanced_prompt = create_enhanced_image_prompt(prompt, child_name, theme)
    logger.info(f"Generating image with Stability AI for {child_name}")
    
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
            "strength": strength,
            "mode": "image-to-image"
        }
        
        # Process the face image
        try:
            # Remove data URL prefix if present
            if "," in face_image_b64:
                face_image_b64 = face_image_b64.split(",")[1]
                
            # Convert base64 to bytes
            image_data = base64.b64decode(face_image_b64)
            
            # Create a temporary file for the image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_file.write(image_data)
                temp_path = temp_file.name
            
            try:
                # Open and process the image
                with Image.open(temp_path) as img:
                    img = img.convert("RGB")
                    img_io = io.BytesIO()
                    img.save(img_io, format='PNG')
                    img_io.seek(0)
                    
                    files = {
                        "image": ("input.png", img_io, "image/png"),
                        "none": ''
                    }
                    
                    # Make the API request
                    response = requests.post(
                        f"{STABILITY_API_HOST}/v2beta/stable-image/generate/sd3",
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=60
                    )
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            return {"error": f"Error processing face image: {str(e)}"}
        
        # Check response
        if response.status_code != 200:
            return {"error": f"Stability API error: {response.status_code} - {response.text}"}
        
        # Return the generated image as base64
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        return {"image": f"data:image/png;base64,{image_base64}", "source": "stability-ai"}
        
    except Exception as e:
        logger.error(f"Stability AI error: {str(e)}")
        return {"error": f"Stability AI error: {str(e)}"}

async def generate_image_with_replicate(
    prompt: str, 
    child_name: str, 
    face_image_b64: str,
    theme: str = "pirate",
    strength: float = 0.7,
    width: int = 1024,
    height: int = 1024
) -> Dict[str, Any]:
    """
    Generate image using Replicate with image-to-image transformation.
    
    Args:
        prompt: Text prompt for image generation
        child_name: Name of the child character
        face_image_b64: Base64-encoded face image
        theme: Theme of the story
        strength: Transformation strength (0.0 to 1.0)
        width: Output image width
        height: Output image height
    
    Returns:
        Dictionary with 'image' (base64) or 'error' key
    """
    if not REPLICATE_API_KEY:
        return {"error": "Replicate API key not configured"}
        
    enhanced_prompt = create_enhanced_image_prompt(prompt, child_name, theme)
    logger.info(f"Generating image with Replicate for {child_name}")
    
    temp_path = None
    try:
        # Save face image to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            # Remove data URL prefix if present
            image_data = base64.b64decode(face_image_b64.split(",")[-1])
            temp_file.write(image_data)
            temp_path = temp_file.name
        
        try:
            # Process the image
            with Image.open(temp_path) as img:
                img = img.convert("RGB")
                img.save(temp_path, "PNG")
            
            # Prepare input parameters for SDXL
            input_params = {
                "prompt": enhanced_prompt,
                "image": open(temp_path, "rb"),
                "width": width,
                "height": height,
                "num_outputs": 1,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "prompt_strength": strength,
                "scheduler": "K_EULER",
                "negative_prompt": "blurry, low quality, distorted, disfigured, extra limbs, extra fingers, cropped, out of frame, watermark, signature, text, scary, dark, frightening"
            }
            
            # Call Replicate API
            output = await asyncio.to_thread(
                replicate.run,
                "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input=input_params
            )
            
            if not output or not isinstance(output, list) or not output[0]:
                return {"error": "Unexpected output format from Replicate"}
            
            # Download the generated image
            response = requests.get(output[0])
            if response.status_code != 200:
                return {"error": f"Failed to download generated image: {response.status_code}"}
            
            # Convert to base64
            image_base64 = base64.b64encode(response.content).decode("utf-8")
            return {"image": f"data:image/png;base64,{image_base64}", "source": "replicate"}
            
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Replicate error: {str(e)}")
        return {"error": f"Replicate error: {str(e)}"}

async def generate_single_image(
    prompt: str, 
    child_name: str, 
    face_image_b64: str,
    theme: str = "pirate",
    strength: float = 0.7,
    width: int = 1024,
    height: int = 1024
) -> Dict[str, Any]:
    """
    Generate a single image using available services (Stability AI first, then Replicate).
    
    Args:
        prompt: Text prompt for image generation
        child_name: Name of the child character
        face_image_b64: Base64-encoded face image
        theme: Theme of the story
        strength: Transformation strength (0.0 to 1.0)
        width: Output image width
        height: Output image height
    
    Returns:
        Dictionary with 'image' (base64), 'source', or 'error' key
    """
    logger.info(f"Starting image generation for {child_name}")
    
    # Try Stability AI first
    if STABILITY_API_KEY:
        logger.info("Trying Stability AI...")
        result = await generate_image_with_stability_ai(
            prompt, child_name, face_image_b64, theme, strength, width, height
        )
        if "image" in result:
            logger.info("Successfully generated image with Stability AI")
            return result
        else:
            logger.warning(f"Stability AI failed: {result.get('error', 'Unknown error')}")
    
    # Fallback to Replicate
    if REPLICATE_API_KEY:
        logger.info("Trying Replicate as fallback...")
        result = await generate_image_with_replicate(
            prompt, child_name, face_image_b64, theme, strength, width, height
        )
        if "image" in result:
            logger.info("Successfully generated image with Replicate")
            return result
        else:
            logger.warning(f"Replicate failed: {result.get('error', 'Unknown error')}")
    
    return {"error": "All image generation services failed"}

# ================================
# API ENDPOINTS
# ================================

@app.post("/api/generate-story-text")
async def generate_story_text_endpoint(request: GenerateStoryRequest):
    """
    Generate story text only (no images yet).
    
    Args:
        request: Contains child_name and theme
    
    Returns:
        Story with text and image prompts, but no generated images yet
    """
    try:
        logger.info(f"Generating story text for {request.child_name} with theme: {request.theme}")
        
        # Generate story using Groq
        story_data = generate_story_with_groq(request.child_name, request.theme)
        
        if not story_data:
            logger.warning("Groq API failed, using fallback story")
            story_data = create_fallback_story(request.child_name, request.theme)
        
        # Create unique story ID
        story_id = str(uuid.uuid4())
        current_time = time.time()
        
        # Convert to our format
        parts = []
        for page in story_data["pages"]:
            part = StoryPart(
                page_number=page["page_number"],
                text=page["story_text"],
                image_prompt=page["image_prompt"],
                image_status="pending"
            )
            parts.append(part)
        
        # Create story response
        story_response = StoryResponse(
            story_id=story_id,
            title=story_data["title"],
            parts=parts,
            status="text_generated",
            created_at=current_time,
            updated_at=current_time
        )
        
        # Store in memory
        STORY_STORE[story_id] = {
            "story": story_response,
            "child_name": request.child_name,
            "theme": request.theme,
            "total_pages": len(parts),
            "completed_images": 0
        }
        
        logger.info(f"Successfully generated story {story_id} with {len(parts)} pages")
        
        return story_response.dict()
        
    except Exception as e:
        logger.error(f"Error generating story text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating story: {str(e)}")

@app.post("/api/generate-image")
async def generate_image_endpoint(request: GenerateImageRequest):
    """
    Generate image for a specific page of an existing story.
    
    Args:
        request: Contains story_id, page_number, user_face_image, and prompt
    
    Returns:
        Generated image data or error
    """
    try:
        logger.info(f"Generating image for story {request.story_id}, page {request.page_number}")
        
        # Check if story exists
        if request.story_id not in STORY_STORE:
            raise HTTPException(status_code=404, detail="Story not found")
        
        story_data = STORY_STORE[request.story_id]
        story = story_data["story"]
        
        # Find the page
        page_index = request.page_number - 1
        if page_index < 0 or page_index >= len(story.parts):
            raise HTTPException(status_code=400, detail="Invalid page number")
        
        # Update page status to processing
        story.parts[page_index].image_status = "processing"
        story.updated_at = time.time()
        
        # Broadcast update
        await manager.broadcast(request.story_id, {
            "type": "image_update",
            "page_number": request.page_number,
            "status": "processing"
        })
        
        # Generate the image
        result = await generate_single_image(
            prompt=request.prompt,
            child_name=story_data["child_name"],
            face_image_b64=request.user_face_image,
            theme=story_data["theme"]
        )
        
        if "image" in result:
            # Success - update the story
            story.parts[page_index].image_url = result["image"]
            story.parts[page_index].image_status = "completed"
            story_data["completed_images"] += 1
            
            # Check if all images are done
            if story_data["completed_images"] >= story_data["total_pages"]:
                story.status = "completed"
            
            story.updated_at = time.time()
            
            # Broadcast success
            await manager.broadcast(request.story_id, {
                "type": "image_update",
                "page_number": request.page_number,
                "status": "completed",
                "image_url": result["image"],
                "story_completed": story.status == "completed"
            })
            
            logger.info(f"Successfully generated image for page {request.page_number}")
            
            return {
                "success": True,
                "image_url": result["image"],
                "source": result.get("source", "unknown"),
                "page_number": request.page_number
            }
        else:
            # Failed - update status
            error_msg = result.get("error", "Unknown error")
            story.parts[page_index].image_status = "failed"
            story.parts[page_index].error = error_msg
            story.updated_at = time.time()
            
            # Broadcast failure
            await manager.broadcast(request.story_id, {
                "type": "image_update",
                "page_number": request.page_number,
                "status": "failed",
                "error": error_msg
            })
            
            logger.error(f"Failed to generate image for page {request.page_number}: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "page_number": request.page_number
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate_image_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")


# Add this at the end of the file
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)