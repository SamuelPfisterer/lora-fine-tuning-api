import os
import shutil
from pathlib import Path
from PIL import Image
from fastapi import UploadFile
import aiofiles
from typing import List

class ImageProcessor:
    @staticmethod
    async def save_uploaded_files(files: List[UploadFile], input_dir: str) -> None:
        Path(input_dir).mkdir(parents=True, exist_ok=True)
        
        for file in files:
            file_path = os.path.join(input_dir, file.filename)
            async with aiofiles.open(file_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)

    @staticmethod
    def crop_and_resize_image(input_path: str, output_path: str, target_size: int = 1024) -> None:
        with Image.open(input_path) as img:
            width, height = img.size
            
            if width > height:
                left = (width - height) // 2
                top = 0
                right = left + height
                bottom = height
            else:
                top = (height - width) // 2
                left = 0
                bottom = top + width
                right = width
            
            img_cropped = img.crop((left, top, right, bottom))
            img_resized = img_cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
            img_resized.save(output_path, quality=95)

    @staticmethod
    def process_images_directory(input_dir: str, output_dir: str) -> int:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.webp'}
        processed_count = 0
        
        for filename in os.listdir(input_dir):
            if Path(filename).suffix.lower() in supported_formats:
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"processed_{filename}")
                try:
                    ImageProcessor.crop_and_resize_image(input_path, output_path)
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        return processed_count

    @staticmethod
    def create_zip_from_images(folder_path: str, zip_name: str = 'images.zip') -> Path:
        base_name = zip_name.replace('.zip', '')
        shutil.make_archive(base_name, 'zip', folder_path)
        return Path(f"{base_name}.zip").absolute() 