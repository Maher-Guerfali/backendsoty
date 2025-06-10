import os
import json
import logging
import uuid
import aiohttp
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from fastapi import HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable is missing")
    raise ValueError("GROQ_API_KEY environment variable is required")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable is missing")
    raise ValueError("GEMINI_API_KEY environment variable is required")

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
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }

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
                    
                    # Clean and parse the JSON response
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
                            parts=[],
                            status="text_generated"
                        )
                        
                        # Add story parts
                        for part_data in story_data.get("parts", []):
                            story.parts.append(StoryPart(
                                part_number=part_data.get("part_number", 1),
                                text=part_data.get("text", ""),
                                image_prompt=part_data.get("image_prompt", ""),
                                status="pending"
                            ))
                        
                        return story
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse story JSON: {content}")
                        raise Exception(f"Failed to parse story JSON: {str(e)}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error: {str(e)}")
            raise Exception(f"Network error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in generate_story: {str(e)}", exc_info=True)
            raise
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
                    result = await response.json()
                    
                    if "error" in result:
                        raise Exception(f"Groq API error: {result['error']}")
                    
                    # Extract the JSON from the response
                    content = result["choices"][0]["message"]["content"]
                    
                    # Clean and parse the JSON response
                    try:
                        # Sometimes the response might include markdown code blocks
                        if '```json' in content:
                            content = content.split('```json')[1].split('```')[0].strip()
                        elif '```' in content:
                            content = content.split('```')[1].split('```')[0].strip()
                        
                        story_data = json.loads(content)
                        
                        # Create the story object
                        story = GeneratedStory(
                            story_id=str(hash(f"{username}_{theme}")),  # Simple ID generation
                            title=story_data.get("title", f"{username}'s {theme.capitalize()} Adventure"),
                            parts=[
                                StoryPart(
                                    part_number=part["part_number"],
                                    text=part["text"],
                                    image_prompt=part["image_prompt"]
                                )
                                for part in story_data["parts"]
                            ]
                        )
                        
                        return story
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Groq response: {e}")
                        logger.error(f"Response content: {content}")
                        raise Exception("Failed to parse story generation response")
                        
        except Exception as e:
            logger.error(f"Error generating story: {str(e)}")
            raise
    
    async def generate_images(self, story: GeneratedStory, face_image_url: str) -> GeneratedStory:
        """
        Generate images for each part of the story using Stability AI
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Create a list of tasks for concurrent processing
                tasks = []
                
                # Create tasks for each part
                for part in story.parts:
                    payload = {
                        "key": STABILITY_API_KEY,
                        "prompt": part.image_prompt,
                        "init_image": face_image_url,
                        "width": "512",
                        "height": "512",
                        "samples": "1",
                        "num_inference_steps": "30",
                        "guidance_scale": 7.5,
                        "strength": 0.7
                    }
                    
                    # Create a coroutine for each image generation
                    async def generate_image(part: StoryPart, payload: dict):
                        try:
                            async with session.post(
                                self.stability_url,
                                json=payload,
                                headers={"Content-Type": "application/json"},
                                timeout=30  # Add timeout
                            ) as response:
                                if response.status != 200:
                                    logger.error(f"HTTP error {response.status} for part {part.part_number}")
                                    part.status = "failed"
                                    return
                                
                                result = await response.json()
                                
                                if "output" not in result:
                                    logger.error(f"Invalid response for part {part.part_number}: {result}")
                                    part.status = "failed"
                                    return
                                
                                # Update the part with the generated image URL
                                part.image_url = result["output"][0]
                                part.status = "completed"
                                
                        except asyncio.TimeoutError:
                            logger.error(f"Timeout generating image for part {part.part_number}")
                            part.status = "failed"
                        except Exception as e:
                            logger.error(f"Error generating image for part {part.part_number}: {str(e)}")
                            part.status = "failed"
                    
                    # Add the task to our list
                    tasks.append(generate_image(part, payload))
                
                # Run all tasks concurrently with a limit
                # We limit to 3 concurrent tasks to avoid overwhelming the API
                async def process_tasks(tasks, limit=3):
                    semaphore = asyncio.Semaphore(limit)
                    
                    async def process_task(task):
                        async with semaphore:
                            await task
                    
                    await asyncio.gather(*[process_task(task) for task in tasks])
                
                await process_tasks(tasks)
                
                # Update story status based on parts
                if all(part.status == "completed" for part in story.parts):
                    story.status = "completed"
                elif any(part.status == "failed" for part in story.parts):
                    story.status = "completed_with_errors"
                
                return story
                
        except Exception as e:
            logger.error(f"Error in generate_images: {str(e)}")
            story.status = "failed"
            return story
