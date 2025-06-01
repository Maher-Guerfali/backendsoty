from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse
from typing import Dict, Any
import uuid
import asyncio
import os

from app.services.story_service import StoryGenerator, StoryPart, GeneratedStory
from pydantic import BaseModel

router = APIRouter()

# In-memory storage for stories (in production, use a database)
stories: Dict[str, GeneratedStory] = {}

class StoryRequest(BaseModel):
    username: str
    theme: str
    face_image_url: str  # Base64 encoded image or URL

@router.post("/generate-story")
async def generate_story(
    request: StoryRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a new story with the given username and theme.
    This will start the generation process in the background.
    """
    try:
        # Validate inputs
        if not request.username or not request.theme:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Username and theme are required",
                    "timestamp": datetime.now().isoformat()
                }
            )

        # Create a unique story ID
        story_id = str(uuid.uuid4())
        
        # Initialize story with pending status
        story = GeneratedStory(
            story_id=story_id,
            title=f"{request.username}'s {request.theme.capitalize()} Adventure",
            parts=[],
            status="pending"
        )
        
        # Store the story
        stories[story_id] = story
        
        # Start the generation process in the background
        background_tasks.add_task(
            generate_story_task,
            story_id=story_id,
            username=request.username,
            theme=request.theme,
            face_image_url=request.face_image_url
        )
        
        logger.info(f"Started story generation for {story_id}")

        return {
            "story_id": story_id,
            "status": "generating_text",
            "created_at": datetime.now().isoformat(),
            "parts": []  # Return empty parts array initially
        }

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
            raise HTTPException(status_code=404, detail="Story not found")
        
        story = stories[story_id]
        
        # Convert the story to a dictionary
        story_dict = story.dict()
        
        # Add some additional status information
        story_dict["completed_parts"] = sum(1 for part in story.parts if part.status == "completed")
        story_dict["total_parts"] = len(story.parts)
        
        return story_dict
    except HTTPException as e:
        logger.error(f"HTTP Error getting story {story_id}: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error getting story {story_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to get story",
                "error": str(e),
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
