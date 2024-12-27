from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    replicate_api_token: str
    huggingface_token: str
    base_model_path: str = "SamuelPfisterer1/hitch/user-lora"
    
    class Config:
        env_file = ".env"

settings = Settings() 