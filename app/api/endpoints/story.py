from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import uuid
import asyncio
import base64
from io import BytesIO
from PIL import Image
import os
import base64
import logging
from datetime import datetime

from app.services.story_service import StoryGenerator, StoryPart, GeneratedStory
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for stories (in production, use a database)
stories: Dict[str, GeneratedStory] = {}

async def generate_story_background(
    story_id: str,
    username: str,
    theme: str,
    face_image_url: str
):
    """
    Background task to generate story and images
    """
    logger.info(f"[Story {story_id}] Starting story generation for {username} with theme: {theme}")
    
    try:
        if story_id not in stories:
            logger.error(f"[Story {story_id}] Story not found in memory")
            return
            
        story = stories[story_id]
        story_generator = StoryGenerator()
        
        try:
            # Generate story text
            logger.info(f"[Story {story_id}] Generating story text...")
            generated_story = await story_generator.generate_story(username, theme)
            
            # Update story with generated parts
            story.title = generated_story.title
            story.parts = generated_story.parts
            story.status = "text_generated"
            story.updated_at = datetime.now().isoformat()
            logger.info(f"[Story {story_id}] Story text generated successfully")
            
            # Generate images if we have a face image
            if face_image_url:
                logger.info(f"[Story {story_id}] Starting image generation with face image")
                try:
                    await story_generator.generate_images(story, face_image_url)
                    story.status = "completed"
                    logger.info(f"[Story {story_id}] Image generation completed successfully")
                except Exception as img_error:
                    error_msg = f"Image generation failed: {str(img_error)}"
                    logger.error(f"[Story {story_id}] {error_msg}", exc_info=True)
                    story.status = "failed"
                    story.error = error_msg
                    # Don't re-raise to prevent the task from failing completely
            else:
                logger.warning(f"[Story {story_id}] No face image provided, skipping image generation")
                story.status = "completed_no_images"
                
        except Exception as gen_error:
            error_msg = f"Story generation failed: {str(gen_error)}"
            logger.error(f"[Story {story_id}] {error_msg}", exc_info=True)
            story.status = "failed"
            story.error = error_msg
            raise  # Re-raise to mark task as failed
            
    except Exception as e:
        error_msg = f"Unexpected error in background task: {str(e)}"
        logger.error(f"[Story {story_id}] {error_msg}", exc_info=True)
        if story_id in stories:
            stories[story_id].status = "failed"
            stories[story_id].error = error_msg
        raise  # Re-raise to mark task as failed
    finally:
        if story_id in stories:
            stories[story_id].updated_at = datetime.now().isoformat()
            logger.info(f"[Story {story_id}] Background task completed with status: {stories[story_id].status}")

router = APIRouter()

# In-memory storage for stories (in production, use a database)
stories: Dict[str, GeneratedStory] = {}

class StoryRequest(BaseModel):
    username: str
    theme: str
    face_image_url: Optional[str] = None  # Optional base64 encoded image or URL

@router.post("/generate-story")
async def generate_story(
    request: Request,
    story_request: StoryRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a new story with the given username and theme.
    This will start the generation process in the background.
    """
    logger.info(f"Received request to generate story: {story_request}")
    
    try:
        # Log request details for debugging
        logger.info(f"Request headers: {dict(request.headers)}")
        
        # Validate inputs
        if not story_request.username or not story_request.theme:
            error_msg = "Username and theme are required"
            logger.error(error_msg)
            raise HTTPException(
                status_code=400,
                detail={
                    "message": error_msg,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        if not story_request.face_image_url:
            error_msg = "Face image is required"
            logger.error(error_msg)
            raise HTTPException(
                status_code=400,
                detail={
                    "message": error_msg,
                    "timestamp": datetime.now().isoformat()
                }
            )

        # Log basic info about the face image
        face_image_info = ""
        if story_request.face_image_url.startswith('data:image'):
            face_image_info = f"Base64 image (length: {len(story_request.face_image_url)})"
        else:
            face_image_info = f"Image URL: {story_request.face_image_url[:100]}..."
            
        logger.info(f"Processing story generation with face image: {face_image_info}")
        face_image_data = None
        if story_request.face_image_url:
            logger.info("Processing face image from request")
            try:
                # Remove data URL prefix if present
                if story_request.face_image_url.startswith('data:'):
                    logger.debug("Extracting base64 from data URL")
                    _, encoded = story_request.face_image_url.split(',', 1)
                    face_image_data = encoded
                else:
                    logger.debug("Using provided URL directly")
                    face_image_data = story_request.face_image_url

                # Validate base64
                if not face_image_data.startswith('http'):
                    logger.debug("Validating base64 data")
                    try:
                        base64.b64decode(face_image_data, validate=True)
                    except Exception as e:
                        raise ValueError(f"Invalid base64 data: {str(e)}")
                    
            except Exception as e:
                error_msg = f"Invalid image data: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": error_msg,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                )
        else:
            logger.info("No face image provided, continuing without it")

        # Create a unique story ID
        story_id = str(uuid.uuid4())
        logger.info(f"Generated story ID: {story_id}")
        
        # Validate face image
        try:
# Test if we can decode the image
            if story_request.face_image_url.startswith('data:image'):
                # Extract the base64 part
                if ',' in story_request.face_image_url:
                    base64_str = story_request.face_image_url.split(',', 1)[1]
                else:
                    base64_str = story_request.face_image_url
                
                # Decode and verify image
                try:
                    img_data = base64.b64decode(base64_str)
                    img = Image.open(BytesIO(img_data))
                    img.verify()  # Verify it's a valid image
                    logger.info(f"Successfully validated base64 image: {img.format}, {img.size}")
                except Exception as img_err:
                    raise ValueError(f"Invalid image data: {str(img_err)}")
            # For URL, we'll validate it in the background task
            elif story_request.face_image_url.startswith('http'):
                logger.info("Face image is a URL, will be validated in background task")
            
        except Exception as img_error:
            error_msg = f"Invalid image format: {str(img_error)}"
            logger.error(error_msg, exc_info=True)
            raise HTTPException(
                status_code=400,
                detail={
                    "message": error_msg,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Create a new story
        story = GeneratedStory(
            story_id=story_id,
            title=f"{story_request.username}'s {story_request.theme.capitalize()} Adventure",
            parts=[],
            status="pending",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        stories[story_id] = story
        
        # Log the start of background task
        logger.info(f"Starting background story generation with ID: {story_id}")
        
        # Start background task with error handling
        try:
            # Create task and store it
            task = background_tasks.add_task(
                generate_story_background,
                story_id=story_id,
                username=story_request.username,
                theme=story_request.theme,
                face_image_url=story_request.face_image_url
            )
            
            # Store task reference in the story for potential cancellation
            story.background_task_id = id(task)
            logger.info(f"Background task {story.background_task_id} started for story ID: {story_id}")
            
            return {
                "status": "started",
                "story_id": story_id,
                "message": "Story generation started",
                "timestamp": datetime.now().isoformat(),
                "details": {
                    "theme": story_request.theme,
                    "username": story_request.username,
                    "has_face_image": bool(story_request.face_image_url)
                }
            }
            
        except Exception as task_error:
            error_msg = f"Failed to start story generation: {str(task_error)}"
            logger.error(error_msg, exc_info=True)
            
            # Clean up the story entry
            if story_id in stories:
                del stories[story_id]
                
            raise HTTPException(
                status_code=500,
                detail={
                    "message": error_msg,
                    "timestamp": datetime.now().isoformat(),
                    "error_type": type(task_error).__name__
                }
            )
        
    except HTTPException as e:
        logger.error(f"HTTP Error generating story: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "message": str(e.detail),
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Error generating story: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to generate story",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "type": type(e).__name__
            }
        )

@router.get("/story/{story_id}", response_model=Dict[str, Any])
async def get_story(story_id: str):
    """
    Get the current status of a story by its ID
    """
    try:
        if story_id not in stories:
            raise HTTPException(status_code=404, detail={
                "message": "Story not found",
                "story_id": story_id,
                "timestamp": datetime.now().isoformat()
            })
        
        story = stories[story_id]
        
        # Convert the story to a dictionary
        story_dict = story.dict()
        
        # Add some additional status information
        completed_parts = sum(1 for part in story.parts if part.status == "completed")
        total_parts = len(story.parts)
        
        # Calculate progress
        progress = 0
        if story.status == "completed":
            progress = 100
        elif story.status == "failed":
            progress = 0
        elif total_parts > 0:
            progress = min(99, int((completed_parts / total_parts) * 100))
        
        # Add metadata
        story_dict.update({
            "completed_parts": completed_parts,
            "total_parts": total_parts,
            "progress": progress,
            "is_complete": story.status in ["completed", "completed_no_images"],
            "is_failed": story.status == "failed",
            "error": getattr(story, "error", None),
            "background_task_id": getattr(story, "background_task_id", None),
            "created_at": getattr(story, "created_at", ""),
            "updated_at": getattr(story, "updated_at", "")
        })
        
        return story_dict
    except HTTPException as e:
        logger.error(f"HTTP Error getting story {story_id}: {str(e)}", exc_info=True)
        if not isinstance(e.detail, dict):
            e.detail = {
                "message": str(e.detail),
                "story_id": story_id,
                "timestamp": datetime.now().isoformat()
            }
        raise
    except Exception as e:
        logger.error(f"Error getting story {story_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to get story",
                "error": str(e),
                "story_id": story_id,
                "timestamp": datetime.now().isoformat()
            }
        )

@router.websocket("/ws/{story_id}")
async def websocket_endpoint(websocket: WebSocket, story_id: str):
    """
    WebSocket endpoint for real-time story updates
    """
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection established for story {story_id}")
        
        while True:
            try:
                # Get the current story status
                if story_id not in stories:
                    logger.warning(f"Story {story_id} not found in WebSocket connection")
                    await websocket.close()
                    break
                
                story = stories[story_id]
                
                # Send the current story status
                await websocket.send_json({
                    "story_id": story_id,
                    "status": story.status,
                    "parts": [
                        {
                            "text": part.text,
                            "image_prompt": part.image_prompt,
                            "image_url": part.image_url,
                            "image_status": part.status,
                            "error": part.error
                        }
                        for part in story.parts
                    ]
                })
                logger.debug(f"Sent update for story {story_id}")
                
                # Wait for changes or close
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                logger.info(f"WebSocket connection cancelled for story {story_id}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop for story {story_id}: {str(e)}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket connection error for story {story_id}: {str(e)}", exc_info=True)
    finally:
        await websocket.close()
        logger.info(f"WebSocket connection closed for story {story_id}")

async def generate_story_task(
    story_id: str,
    username: str,
    theme: str,
    face_image_url: str
):
    """
    Background task to generate a story and its images
    """
    try:
        generator = StoryGenerator()
        
        # Update story status
        if story_id in stories:
            stories[story_id].status = "generating_text"
            logger.info(f"Starting text generation for story {story_id}")
        else:
            logger.error(f"Story {story_id} not found during generation")
            return
        
        # Step 1: Generate the story text
        try:
            story = await generator.generate_story(username, theme)
            logger.info(f"Successfully generated text for story {story_id}")
            
            # Update story with generated text
            if story_id in stories:
                stories[story_id] = story
            else:
                logger.error(f"Story {story_id} disappeared during text generation")
                return
            
        except Exception as e:
            logger.error(f"Error generating story text for {story_id}: {str(e)}", exc_info=True)
            if story_id in stories:
                stories[story_id].status = "failed"
                stories[story_id].error = str(e)
            return
            
        # Step 2: Generate images if we have a face image
        if face_image_url:
            try:
                logger.info(f"Starting image generation for story {story_id}")
                
                # Generate images concurrently
                tasks = []
                for part in story.parts:
                    async def generate_image(part: StoryPart):
                        try:
                            part.image_url = await generator.generate_image(
                                prompt=part.image_prompt,
                                username=username,
                                face_image_url=face_image_url
                            )
                            part.status = "completed"
                            logger.info(f"Successfully generated image for part {part.part_number} of story {story_id}")
                        except Exception as img_error:
                            logger.error(f"Error generating image for part {part.part_number} of story {story_id}: {str(img_error)}")
                            part.status = "failed"
                            part.error = str(img_error)
                    tasks.append(generate_image(part))
                
                await asyncio.gather(*tasks)
                
                # Update story status
                if story_id in stories:
                    stories[story_id].status = "completed"
                    logger.info(f"Story {story_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Error during image generation for story {story_id}: {str(e)}", exc_info=True)
                if story_id in stories:
                    stories[story_id].status = "failed"
                    stories[story_id].error = str(e)
    except Exception as e:
        logger.error(f"Fatal error in story generation for {story_id}: {str(e)}", exc_info=True)
        if story_id in stories:
            stories[story_id].status = "failed"
            stories[story_id].error = str(e)
        
        # Update the story in storage
        if story_id in stories:
            stories[story_id].title = story.title
            stories[story_id].parts = story.parts
            stories[story_id].status = "generating_images"
        
        # Step 2: Generate images for each part
        if face_image_url:
            story = await generator.generate_images(story, face_image_url)
            
            # Update the story with generated images
            if story_id in stories:
                stories[story_id].parts = story.parts
                stories[story_id].status = story.status
        
    except Exception as e:
        if story_id in stories:
            stories[story_id].status = "failed"
            stories[story_id].error = str(e)
        # In a production environment, you might want to log this error to a monitoring service
        print(f"Error generating story {story_id}: {str(e)}")
    
    return {"status": "completed" if story_id not in stories or stories[story_id].status != "failed" else "failed"}
