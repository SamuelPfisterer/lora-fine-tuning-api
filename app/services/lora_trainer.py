import replicate
from pathlib import Path
from typing import Optional

class LoraTrainer:
    @staticmethod
    def create_user_model(user_id: str) -> None:
        try:
            model = replicate.models.create(
                owner="samuelpfisterer",
                name=f"hitch-lora-{user_id}",
                visibility="private",
                hardware="gpu-a100-large"
            )
            print(f"Created Replicate model: samuelpfisterer/hitch-lora-{user_id}")
            return model
        except Exception as e:
            if "already exists" in str(e):
                print(f"Model samuelpfisterer/hitch-lora-{user_id} already exists")
                return None
            raise e

    @staticmethod
    def train_lora(
        zip_file_path: Path, 
        hf_token: str, 
        hf_model_path: str, 
        user_id: str,
        webhook_url: Optional[str] = None
    ):
        try:
            full_model_path = f"{hf_model_path}/{user_id}"
            LoraTrainer.create_user_model(user_id)
            
            training_config = {
                "version": "ostris/flux-dev-lora-trainer:1296f0ab2d695af5a1b5eeee6e8ec043145bef33f1675ce1a2cdb0f81ec43f02",
                "destination": f"samuelpfisterer/hitch-lora-{user_id}",
                "input": {
                    "input_images": None,  # Will be set below
                    "lora_rank": 16,
                    "resolution": "512,768,1024",
                    "autocaption": True,
                    "trigger_word": "HKC",
                    "steps": 2000,
                    "batch_size": 1,
                    "learning_rate": 0.0004,
                    "optimizer": "adamw8bit",
                    "wandb_project": "flux_train_replicate",
                    "wandb_save_interval": 100,
                    "caption_dropout_rate": 0.05,
                    "cache_latents_to_disk": False,
                    "wandb_sample_interval": 100
                }
            }

            # Add webhook configuration if URL is provided
            if webhook_url:
                training_config["webhook"] = webhook_url
                training_config["webhook_events_filter"] = ["start", "completed"]
            
            with open(zip_file_path, "rb") as file_input:
                training_config["input"]["input_images"] = file_input
                training = replicate.trainings.create(**training_config)
                return training
                
        except Exception as e:
            print(f"Error during training setup: {str(e)}")
            raise 