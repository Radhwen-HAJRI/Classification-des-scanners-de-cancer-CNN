import zipfile
import gdown
from src.CNNClassifier.utils.common import get_size
from src.CNNClassifier.log_config import logging
from src.CNNClassifier.entity.config_entity import DataIngestionConfig
import os

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_data(self) -> str:
        try:
            dataset_url = self.config.source_URL
            zip_file_path = self.config.local_data_file
            os.makedirs(self.config.root_dir, exist_ok=True)
            logging.info(f"Downloading file from :[{dataset_url}] into :[{zip_file_path}]")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            download_url = prefix + file_id
            gdown.download(url=download_url, output=str(zip_file_path), quiet=False)


            logging.info(f"File downloaded successfully and saved at :[{zip_file_path}]")
        except Exception as e:
            raise e
    def extract_zip_file(self):
        try:
            unzip_path = self.config.unzip_dir

            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            
        except Exception as e:
            raise e