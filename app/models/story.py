from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class StoryPart(BaseModel):
    part_number: int
    text: str
    image_prompt: str
    image_url: Optional[str] = None
    status: str = "pending"
    error: Optional[str] = None

class Story(Base):
    __tablename__ = "stories"

    id = Column(Integer, primary_key=True, index=True)
    story_id = Column(String, unique=True, index=True)
    title = Column(String)
    child_name = Column(String)
    theme = Column(String)
    status = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    data = Column(JSON)  # Store parts and other data as JSON

    def to_dict(self):
        return {
            "story_id": self.story_id,
            "title": self.title,
            "child_name": self.child_name,
            "theme": self.theme,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "parts": self.data.get("parts", [])
        }
