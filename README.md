# LoRA Fine-Tuning API

This FastAPI application provides an endpoint for fine-tuning LoRA models using Replicate. It processes uploaded images and initiates training for custom LoRA models.

## Features

- Image processing (cropping and resizing)
- Automatic LoRA model training on Replicate
- Asynchronous file handling
- Automatic cleanup of temporary files
- Environment-based configuration

## Prerequisites

- Python 3.8+
- Replicate API token
- Hugging Face token
- Sufficient disk space for temporary image processing

## Installation

1. Clone the repository: 

I'll create a comprehensive README.md that explains the API setup, usage, and functionality.

```markdown:README.md
# LoRA Fine-Tuning API

This FastAPI application provides an endpoint for fine-tuning LoRA models using Replicate. It processes uploaded images and initiates training for custom LoRA models.

## Features

- Image processing (cropping and resizing)
- Automatic LoRA model training on Replicate
- Asynchronous file handling
- Automatic cleanup of temporary files
- Environment-based configuration

## Prerequisites

- Python 3.8+
- Replicate API token
- Hugging Face token
- Sufficient disk space for temporary image processing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your credentials:
```env
REPLICATE_API_TOKEN=your_replicate_token
HUGGINGFACE_TOKEN=your_huggingface_token
BASE_MODEL_PATH=SamuelPfisterer1/hitch/user-lora
```

## Running the API

Start the server with:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Documentation

### Endpoint: POST /train-lora/

Initiates LoRA model training with provided images.

#### Request

- Method: `POST`
- Content-Type: `multipart/form-data`
- Query Parameters:
  - `user_id`: String identifier for the user (required)
    - Length: 1-50 characters
    - Pattern: Must match `^[a-z0-9_-]+[a-z0-9_]$`
    - Description: ID of the user that is also used to generate photos of the user
  - `webhook_url`: URL for training status notifications (optional)
    - Example: `https://example.com/webhook`
    - Description: URL for Webhook notifications, e.g., when the training finishes
- Request Body (multipart/form-data):
  - `files`: List of image files (required)


Example using curl:
```bash
curl -X POST "http://localhost:8000/train-lora/?user_id=test-user_123&webhook_url=https://example.com/webhook"
-H "Content-Type: multipart/form-data"
-F "files=@image1.jpg"
-F "files=@image2.jpg"
```

Example using Python requests:
```python
import requests
files = [
('files', ('image1.jpg', open('image1.jpg', 'rb'))),
('files', ('image2.jpg', open('image2.jpg', 'rb')))
]
response = requests.post(
'https://lora-fine-tuning-api.onrender.com/train-lora/',
params={
  'user_id': 'test-user_123', # Must match pattern: ^[a-z0-9_-]+[a-z0-9_]$
  'webhook_url': 'https://example.com/webhook'
},
files=files
)
```

#### Response

```json
{
    "training_id": "string",
    "status": "string",
    "huggingface_path": "string",
    "replicate_model": "string"
}
```

### Webhook Notifications

When a webhook URL is provided, the API will send POST requests with JSON payloads for the following events:
- Training start: `status: "starting"`
- Training completion: `status: "successful"`

Example webhook payload:
```json
{
    "training_id": "gpbxnb8z6drma0cm1tmardb0mr",
    "status": "starting",
    "huggingface_path": "SamuelPfisterer1/hitch/user-lora/webhook_test_user",
    "replicate_model": "samuelpfisterer/hitch-lora-webhook_test_user"
}
```

#### Webhook Implementation Tips

1. **User ID Extraction**: You can extract the user_id from the `replicate_model` field:
   - Format: `samuelpfisterer/hitch-lora-{user_id}`
   - Example: For model `samuelpfisterer/hitch-lora-webhook_test_user`, the user_id is `webhook_test_user`

2. **Recommended Webhook URL Pattern**:
   Instead of using a single webhook URL, consider using path parameters to handle different users:
   ```
   https://your-domain.com/webhooks/{user_id}/lora-training
   ```
   This way, you can automatically route notifications to the correct user handler.

Example webhook handler in Python:
```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/webhooks/{user_id}/lora-training")
async def handle_training_webhook(user_id: str, payload: dict):
    # Extract status
    status = payload.get("status")
    
    # Verify user_id matches the one in replicate_model
    model_name = payload.get("replicate_model", "")
    expected_suffix = f"hitch-lora-{user_id}"
    if not model_name.endswith(expected_suffix):
        raise HTTPException(status_code=400, detail="User ID mismatch")
    
    if status == "starting":
        # Handle training start
        pass
    elif status == "successful":
        # Handle training completion
        pass
    
    return {"status": "processed"}

## How It Works

1. **Image Upload**: The API accepts multiple image files through a POST request.

2. **Image Processing**:
   - Images are saved to a temporary directory
   - Each image is cropped to a square aspect ratio
   - Images are resized to a standard size (1024x1024)
   - Processed images are saved to a separate temporary directory

3. **ZIP Creation**: Processed images are compressed into a ZIP file for training

4. **Model Creation**: A new model is created on Replicate with the user's ID

5. **Training Initiation**: The ZIP file is uploaded to Replicate and training begins

6. **Cleanup**: All temporary files are automatically removed after the request completes

## Error Handling

The API includes error handling for:
- Missing or invalid files
- Image processing failures
- Training initialization failures
- File system operations

## Development

The project structure is organized as follows:
```
.
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application and routes
│   ├── config.py         # Configuration management
│   ├── services/
│   │   ├── __init__.py
│   │   ├── image_processor.py  # Image processing logic
│   │   └── lora_trainer.py     # LoRA training logic
│   └── schemas/
│       ├── __init__.py
│       └── requests.py    # Pydantic models
├── requirements.txt
└── README.md
```

## Deployment

This API can be deployed on any platform that supports Python web applications. For production deployment:

1. Set environment variables instead of using `.env`
2. Use a production ASGI server like Gunicorn with Uvicorn workers
3. Configure appropriate security measures (API keys, rate limiting, etc.)
