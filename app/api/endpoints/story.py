from fastapi import APIRouter, HTTPException, BackgroundTasks
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

@router.post("/generate-story", response_model=Dict[str, Any])
async def create_story(
    request: StoryRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a new story with the given username and theme.
    This will start the generation process in the background.
    """
    try:
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
        
        return {
            "status": "started",
            "story_id": story_id,
            "message": "Story generation started. Check status using /story/{story_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/story/{story_id}", response_model=Dict[str, Any])
async def get_story(story_id: str):
    """
    Get the current status of a story by its ID
    """
    if story_id not in stories:
        raise HTTPException(status_code=404, detail="Story not found")
    
    story = stories[story_id]
    
    # Convert the story to a dictionary
    story_dict = story.dict()
    
    # Add some additional status information
    story_dict["completed_parts"] = sum(1 for part in story.parts if part.status == "completed")
    story_dict["total_parts"] = len(story.parts)
    
    return story_dict

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
        
        # Step 1: Generate the story text
        story = await generator.generate_story(username, theme)
        
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
