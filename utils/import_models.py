import kagglehub
from pathlib import Path


class ModelImporter:
    def __init__(self):
        self.models_path = Path("../models").resolve()
        self.models_path.mkdir(exist_ok=True)
        self.kagglehub_link = "blaku03/player-detection/pyTorch"

    def download_kaggle_model(self, model_name: str) -> str:
        saved_path = kagglehub.model_download(f"{self.kagglehub_link}/{model_name}")
        print(f"Model downloaded to {saved_path}")
        return saved_path

    def download_field_model(self, model_name: str = "field_recognitionv1_0") -> str:
        print("Downloading field recognition model...")
        return self.download_kaggle_model(model_name)

    def download_player_model(self, model_name: str = "player_detectionv1_1") -> str:
        print("Downloading player detection model...")
        return self.download_kaggle_model(model_name)
