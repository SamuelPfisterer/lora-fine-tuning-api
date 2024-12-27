import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import uuid
from app.config import settings
from app.services.image_processor import ImageProcessor
from app.services.lora_trainer import LoraTrainer
from app.schemas.requests import LoraTrainingRequest, LoraTrainingResponse

app = FastAPI()

@app.post("/train-lora/", response_model=LoraTrainingResponse)
async def train_lora(
    files: List[UploadFile] = File(...),
    user_id: str = None,
    webhook_url: str = None
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Create temporary directories with unique names
    session_id = str(uuid.uuid4())
    input_dir = f"temp_input_{session_id}"
    processed_dir = f"temp_processed_{session_id}"
    
    try:
        # Save uploaded files
        await ImageProcessor.save_uploaded_files(files, input_dir)
        
        # Process images
        processed_count = ImageProcessor.process_images_directory(input_dir, processed_dir)
        if processed_count == 0:
            raise HTTPException(status_code=400, detail="No valid images were processed")
        
        # Create zip file
        zip_file_path = ImageProcessor.create_zip_from_images(processed_dir)
        
        # Start training
        training = LoraTrainer.train_lora(
            zip_file_path,
            settings.huggingface_token,
            settings.base_model_path,
            user_id,
            webhook_url
        )
        
        return LoraTrainingResponse(
            training_id=training.id,
            status=training.status,
            huggingface_path=f"{settings.base_model_path}/{user_id}",
            replicate_model=f"samuelpfisterer/hitch-lora-{user_id}"
        )
        
    finally:
        # Cleanup
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(processed_dir, ignore_errors=True)
        if 'zip_file_path' in locals():
            os.remove(zip_file_path) 