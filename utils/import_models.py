from typing import Optional
import kagglehub
from pathlib import Path


class ModelImporter:
    def __init__(self, base_kagglehub_link: str = "blaku03/player-detection/pyTorch"):
        self.kagglehub_link = base_kagglehub_link

    def download_kaggle_model(
        self, model_name: str, version: Optional[int] = None
    ) -> str:
        version_suffix = f"/{version}" if version else ""
        saved_path = kagglehub.model_download(
            f"{self.kagglehub_link}/{model_name}{version_suffix}"
        )
        print(f"Model downloaded to {saved_path}")

        # Convert to Path object
        path = Path(saved_path)

        # Find all .pt files directly in the downloaded directory
        pt_files = list(path.glob("*.pt"))

        if pt_files:
            # Return the path to the first .pt file found
            model_path = str(pt_files[0])
            print(f"Model file location at {model_path}")
            return model_path
        else:
            print(f"Warning: No .pt file found in {saved_path}")
            return saved_path

    def download_pitch_model(self, version: Optional[int] = None) -> str:
        model_name = "pitch_detection"
        print("Downloading field recognition model...")
        return self.download_kaggle_model(model_name=model_name, version=version)

    def download_player_model(self, version: Optional[int] = None) -> str:
        model_name = "player_detection"
        print("Downloading player detection model...")
        return self.download_kaggle_model(model_name=model_name, version=version)
