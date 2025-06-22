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
import replicate
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Get environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Validate required environment variables
if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    logger.error("GROQ_API_KEY environment variable is missing or not configured")
    raise ValueError("GROQ_API_KEY environment variable is required")

if not REPLICATE_API_TOKEN or REPLICATE_API_TOKEN == "your_replicate_api_token_here":
    logger.error("REPLICATE_API_TOKEN environment variable is missing or not configured")
    raise ValueError("REPLICATE_API_TOKEN environment variable is required")

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
    status: str = "text_generated"

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
        Generate a 4-part story with image prompts using Groq API
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
            
            For each part, provide:
            1. A story segment (about 100 words)
            2. A detailed image prompt that visually represents that part of the story
            
            Format the response as a JSON object with this structure:
            {{
                "title": "Story Title",
                "parts": [
                    {{
                        "part_number": 1,
                        "text": "Story text for part 1...",
                        "image_prompt": "Detailed image description for part 1..."
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
                    "max_tokens": 4000
                }
                
                async with session.post(
                    self.groq_url, 
                    headers=self.headers, 
                    json=data
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
                        
                        # Create the story object
                        story = GeneratedStory(
                            story_id=str(uuid.uuid4()),
                            title=story_data.get("title", f"{username}'s {theme.capitalize()} Adventure"),
                            parts=[
                                StoryPart(
                                    part_number=part.get("part_number", idx + 1),
                                    text=part.get("text", ""),
                                    image_prompt=part.get("image_prompt", ""),
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
        Generate images for each part of the story using Replicate's img2img
        """
        if not REPLICATE_API_TOKEN:
            raise HTTPException(
                status_code=500,
                detail="Replicate API token not configured"
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
            
            # Process each story part
            for part in story.parts:
                temp_img_path = None
                try:
                    logger.info(f"Generating image for part {part.part_number}")
                    
                    # Create a temporary file for the face image
                    temp_img_path = f"temp_face_{uuid.uuid4()}.png"
                    face_image.save(temp_img_path, format='PNG')
                    
                    # First try: img2img
                    try:
                        output = replicate.run(
                            "stability-ai/stable-diffusion-img2img:15a3689ee13b0d2616e98820eca31d4c3abcd36672df6afce5cb6feb1d66087d",
                            input={
                                "image": open(temp_img_path, "rb"),
                                "prompt": part.image_prompt,
                                "num_inference_steps": 30,
                                "guidance_scale": 7.5,
                                "strength": 0.8,
                                "negative_prompt": "blurry, low quality, distorted, deformed, text, watermark"
                            }
                        )
                        
                        # Get the generated image URL
                        image_url = output[0] if isinstance(output, list) else output
                        
                        # Download the generated image
                        response = requests.get(image_url, timeout=60)
                        response.raise_for_status()
                        
                        # Convert to base64 for storage
                        img_data = io.BytesIO(response.content)
                        part.image_url = f"data:image/png;base64,{base64.b64encode(img_data.getvalue()).decode()}"
                        part.status = "completed"
                        logger.info(f"Successfully generated image for part {part.part_number}")
                        
                    except Exception as img2img_error:
                        logger.warning(f"Image generation with img2img failed, trying txt2img: {str(img2img_error)}")
                        
                        # Fallback: txt2img
                        try:
                            output = replicate.run(
                                "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
                                input={
                                    "prompt": f"{part.image_prompt}, high quality, detailed, professional",
                                    "width": 512,
                                    "height": 512,
                                    "num_inference_steps": 30,
                                    "guidance_scale": 7.5,
                                    "negative_prompt": "blurry, low quality, distorted, deformed, text, watermark"
                                }
                            )
                            
                            # Get the generated image URL
                            image_url = output[0] if isinstance(output, list) else output
                            
                            # Download the generated image
                            response = requests.get(image_url, timeout=60)
                            response.raise_for_status()
                            
                            # Convert to base64 for storage
                            img_data = io.BytesIO(response.content)
                            part.image_url = f"data:image/png;base64,{base64.b64encode(img_data.getvalue()).decode()}"
                            part.status = "completed"
                            logger.info(f"Successfully generated fallback image for part {part.part_number}")
                            
                        except Exception as txt2img_error:
                            logger.error(f"Failed to generate image with txt2img: {str(txt2img_error)}")
                            part.status = f"error: {str(txt2img_error)}"
                            continue
                            
                except Exception as e:
                    logger.error(f"Unexpected error generating image for part {part.part_number}: {str(e)}")
                    part.status = f"error: {str(e)}"
                    
                finally:
                    # Clean up the temporary file
                    if temp_img_path and os.path.exists(temp_img_path):
                        try:
                            os.remove(temp_img_path)
                        except Exception as e:
                            logger.warning(f"Failed to remove temporary file {temp_img_path}: {str(e)}")
            
            # Update story status based on parts
            if all(part.status == "completed" for part in story.parts):
                story.status = "completed"
            elif any(part.status == "completed" for part in story.parts):
                story.status = "completed_with_errors"
            else:
                story.status = "failed"
            
            return story
                
        except Exception as e:
            logger.error(f"Error in generate_images: {str(e)}", exc_info=True)
            story.status = "failed"
            story.error = str(e)
            return story
