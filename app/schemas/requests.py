from pydantic import BaseModel, HttpUrl
from typing import List
from fastapi import UploadFile, File

class LoraTrainingResponse(BaseModel):
    training_id: str
    status: str
    huggingface_path: str
    replicate_model: str

class LoraTrainingRequest(BaseModel):
    user_id: str
    webhook_url: HttpUrl | None = None  # Optional webhook URL 