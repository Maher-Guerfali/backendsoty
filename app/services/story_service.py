import os
import json
import logging
import uuid
import aiohttp
import base64
import io
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from fastapi import HTTPException
from PIL import Image
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Get environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Validate required environment variables
if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    logger.error("GROQ_API_KEY environment variable is missing or not configured")
    raise ValueError("GROQ_API_KEY environment variable is required")

if not HF_API_TOKEN or HF_API_TOKEN == "your_hf_api_token_here":
    logger.error("HF_API_TOKEN environment variable is missing or not configured")
    raise ValueError("HF_API_TOKEN environment variable is required")

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

class StoryPart(BaseModel):
    part_number: int
    text: str
    image_prompt: str
    image_url: Optional[str] = None
    status: str = "pending"

class GeneratedStory(BaseModel):
    story_id: str
    title: str
    parts: List[StoryPart]
    status: str = "pending"
    error: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    background_task_id: Optional[int] = None

class StoryGenerator:
    def __init__(self):
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
        
    def _decode_base64_image(self, base64_string: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))

    async def generate_story(self, username: str, theme: str) -> GeneratedStory:
        """
        Generate a 4-part story with a single image that represents the overall story
        """
        try:
            # Validate inputs
            if not username or not theme:
                raise ValueError("Username and theme are required")

            # Log the generation attempt
            logger.info(f"Starting story generation for {username} with theme: {theme}")
            
            # Prepare the prompt for Groq
            prompt = f"""
            Create a 4-part children's story with the following details:
            - Main character: {username}
            - Theme: {theme}
            
            Also provide a single, detailed image prompt that visually represents the overall story.
            The image should be suitable for all parts of the story and should clearly show {username}'s face.
            
            Format the response as a JSON object with this structure:
            {{
                "title": "Story Title",
                "image_prompt": "Detailed image description that represents the whole story...",
                "parts": [
                    {{
                        "part_number": 1,
                        "text": "Story text for part 1..."
                    }},
                    {{
                        "part_number": 2,
                        "text": "Story text for part 2..."
                    }},
                    {{
                        "part_number": 3,
                        "text": "Story text for part 3..."
                    }},
                    {{
                        "part_number": 4,
                        "text": "Story text for part 4..."
                    }}
                ]
            }}
            """
            
            # Call Groq API
            async with aiohttp.ClientSession() as session:
                data = {
                    "model": "mixtral-8x7b-32768",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "response_format": {"type": "json_object"}
                }
                
                async with session.post(
                    self.groq_url, 
                    headers=self.headers, 
                    json=data,
                    timeout=60
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Groq API error {response.status}: {error_text}")
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Failed to generate story text: {error_text}"
                        )
                    
                    result = await response.json()
                    
                    if "error" in result:
                        raise Exception(f"Groq API error: {result['error']}")
                    
                    # Extract the JSON from the response
                    content = result["choices"][0]["message"]["content"]
                    
                    try:
                        # Sometimes the response might include markdown code blocks
                        if '```json' in content:
                            content = content.split('```json')[1].split('```')[0].strip()
                        elif '```' in content:
                            content = content.split('```')[1].strip()
                            if content.startswith('json'):
                                content = content[4:].strip()
                        
                        story_data = json.loads(content)
                        
                        # Get the single image prompt for the whole story
                        overall_image_prompt = story_data.get("image_prompt", "")
                        if not overall_image_prompt:
                            overall_image_prompt = (
                                f"A beautiful illustration showing {username} in a {theme} setting. "
                                f"{username}'s face should be clearly visible. "
                                "Children's book style, vibrant colors, magical atmosphere."
                            )
                        
                        # Create the story object with the same image prompt for all parts
                        story = GeneratedStory(
                            story_id=str(uuid.uuid4()),
                            title=story_data.get("title", f"{username}'s {theme.capitalize()} Adventure"),
                            parts=[
                                StoryPart(
                                    part_number=part.get("part_number", idx + 1),
                                    text=part.get("text", ""),
                                    image_prompt=overall_image_prompt,
                                    status="pending"
                                )
                                for idx, part in enumerate(story_data.get("parts", []))
                            ],
                            status="text_generated"
                        )
                        
                        return story
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse story JSON: {content}")
                        raise Exception(f"Failed to parse story JSON: {str(e)}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error: {str(e)}")
            raise Exception(f"Network error: {str(e)}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq response: {e}")
            logger.error(f"Response content: {content}")
            raise Exception("Failed to parse story generation response")
            
        except Exception as e:
            logger.error(f"Error in generate_story: {str(e)}", exc_info=True)
            raise
    
    async def generate_images(self, story: GeneratedStory, face_image_url: str) -> GeneratedStory:
        """
        Generate a single image for the story and use it for all parts
        """
        if not HF_API_TOKEN:
            raise HTTPException(
                status_code=500,
                detail="Hugging Face API token not configured"
            )
            
        if not face_image_url:
            raise HTTPException(
                status_code=400,
                detail="No face image provided"
            )
            
        try:
            # Download the face image
            try:
                if face_image_url.startswith('http'):
                    response = requests.get(face_image_url, timeout=30)
                    response.raise_for_status()
                    face_image = Image.open(io.BytesIO(response.content))
                elif face_image_url.startswith('data:image'):
                    # Handle base64 encoded image
                    face_image = self._decode_base64_image(face_image_url)
                else:
                    raise ValueError("Invalid image format. Must be a URL or base64 encoded image.")
                
                face_image = face_image.convert("RGB")
                
            except Exception as e:
                logger.error(f"Failed to load face image: {str(e)}")
                story.status = "failed"
                story.error = f"Failed to load face image: {str(e)}"
                return story
            
            # Generate a single image for the entire story
            # Use the first part's prompt or create a general one
            image_prompt = story.parts[0].image_prompt if story.parts else f"{story.title}, children's book illustration"
            
            temp_img_path = None
            generated_image_url = None
            
            try:
                # Create a temporary file for the face image
                temp_img_path = f"temp_face_{uuid.uuid4()}.png"
                face_image.save(temp_img_path, format='PNG')
                
                # Generate image using Hugging Face API
                logger.info(f"Generating story image with prompt: {image_prompt}")
                
                # Enhance the prompt to ensure face is visible
                enhanced_prompt = (
                    f"{image_prompt} "
                    "High quality, detailed, professional illustration, children's book style, "
                    "vibrant colors, magical atmosphere, clear facial features, full body shot"
                )
                
                # Prepare the request data
                with open(temp_img_path, "rb") as img_file:
                    base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                
                data = {
                    "inputs": {
                        "prompt": enhanced_prompt,
                        "init_image": f"data:image/png;base64,{base64_image}",
                        "strength": 0.7,
                        "guidance_scale": 7.5,
                        "num_inference_steps": 40,
                        "negative_prompt": (
                            "blurry, low quality, distorted, deformed, text, watermark, "
                            "multiple faces, cropped faces, bad anatomy, extra limbs"
                        )
                    }
                }
                
                # Call Hugging Face API
                response = requests.post(
                    HF_API_URL,
                    headers=HF_HEADERS,
                    json=data,
                    timeout=120
                )
                
                if response.status_code != 200:
                    error_msg = f"Hugging Face API error {response.status_code}: {response.text}"
                    raise Exception(error_msg)
                
                # Get the generated image
                img_data = io.BytesIO(response.content)
                img_base64 = base64.b64encode(img_data.getvalue()).decode()
                generated_image_url = f"data:image/png;base64,{img_base64}"
                
                # Save a local copy for debugging
                debug_img_path = f"debug_story_image.png"
                with open(debug_img_path, "wb") as f:
                    f.write(base64.b64decode(img_base64))
                logger.info(f"Saved debug image to {debug_img_path}")
                
            except Exception as img_gen_error:
                logger.error(f"Image generation failed: {str(img_gen_error)}")
                story.status = "failed"
                story.error = f"Image generation failed: {str(img_gen_error)}"
                return story
                
            finally:
                # Clean up the temporary file
                if temp_img_path and os.path.exists(temp_img_path):
                    try:
                        os.remove(temp_img_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file {temp_img_path}: {str(e)}")
            
            # If we successfully generated an image, assign it to all story parts
            if generated_image_url:
                for part in story.parts:
                    part.image_url = generated_image_url
                    part.status = "completed"
                story.status = "completed"
            else:
                story.status = "failed"
                story.error = "Failed to generate story image"
            
            return story
                
        except Exception as e:
            logger.error(f"Error in generate_images: {str(e)}", exc_info=True)
            story.status = "failed"
            story.error = str(e)
            return story
