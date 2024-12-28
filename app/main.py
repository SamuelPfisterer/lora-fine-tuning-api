import os
import shutil
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from typing import List, Optional, Annotated
import uuid
import logging
from app.config import settings
from app.services.image_processor import ImageProcessor
from app.services.lora_trainer import LoraTrainer
from app.schemas.requests import LoraTrainingRequest, LoraTrainingResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/train-lora/", response_model=LoraTrainingResponse)
async def train_lora(
    files: Annotated[List[UploadFile], File()],
    user_id: Annotated[str, Query(
    description="ID of the user that is also used to generate photos of the user",
    min_length=1,
    max_length=50,
    pattern="^[a-z0-9_-]+[a-z0-9_]$"
    )],
    webhook_url: Annotated[Optional[str], Query(
        description="URL for Webhook notifications, e.g., when the training finishes",
        example="https://example.com/webhook"
    )] = None
):
    # Log incoming request details
    logger.info(f"Received training request - User ID: {user_id}")
    logger.info(f"Webhook URL: {webhook_url}")
    logger.info(f"Number of files received: {len(files)}")
    logger.info(f"File names: {[file.filename for file in files]}")

    if not files:
        logger.error("No files uploaded in request")
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Create temporary directories with unique names
    session_id = str(uuid.uuid4())
    input_dir = f"temp_input_{session_id}"
    processed_dir = f"temp_processed_{session_id}"
    
    logger.info(f"Created session with ID: {session_id}")
    
    try:
        # Save uploaded files
        logger.info("Saving uploaded files...")
        await ImageProcessor.save_uploaded_files(files, input_dir)
        
        # Process images
        logger.info("Processing images...")
        processed_count = ImageProcessor.process_images_directory(input_dir, processed_dir)
        logger.info(f"Successfully processed {processed_count} images")

        if processed_count == 0:
            logger.error("No valid images were processed")
            raise HTTPException(status_code=400, detail="No valid images were processed")
        
        # Create zip file
        logger.info("Creating ZIP file from processed images...")
        zip_file_path = ImageProcessor.create_zip_from_images(processed_dir)
        logger.info(f"ZIP file created at: {zip_file_path}")
        
        # Start training
        logger.info("Initiating LoRA training...")
        training = LoraTrainer.train_lora(
            zip_file_path,
            settings.huggingface_token,
            settings.base_model_path,
            user_id,
            webhook_url
        )
        
        response = LoraTrainingResponse(
            training_id=training.id,
            status=training.status,
            huggingface_path=f"{settings.base_model_path}/{user_id}",
            replicate_model=f"samuelpfisterer/hitch-lora-{user_id}"
        )
        
        logger.info(f"Training initiated successfully - Training ID: {training.id}")
        logger.info(f"Initial status: {training.status}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during training process: {str(e)}", exc_info=True)
        raise
        
    finally:
        # Cleanup
        logger.info("Cleaning up temporary files...")
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(processed_dir, ignore_errors=True)
        if 'zip_file_path' in locals():
            os.remove(zip_file_path)
        logger.info("Cleanup complete") 