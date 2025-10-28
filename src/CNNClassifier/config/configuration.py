from src.CNNClassifier.constants import *
from src.CNNClassifier.utils.common import read_yaml, create_directories,save_json
from src.CNNClassifier.entity.config_entity import (DataIngestionConfig,
                                                    PrepareBaseModelConfig,TrainingConfig,EvaluationConfig)
from pathlib import Path

import os

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = config_file_path,
        params_filepath = params_file_path):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root])



    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir = Path(config.root_dir),
            source_URL = config.source_URL,
            local_data_file = Path(config.local_data_file),
            unzip_dir = Path(config.unzip_dir)
        )
        
        return data_ingestion_config
    
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:

        config = self.config.prepare_base_model
        create_directories([config.root_dir])
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )
        
        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training =  self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        # CORRECTION : Utilisez le nouveau chemin de préparation des données
        training_data= os.path.join("artifacts", "data_preparation", "kidney-ct-scan-image")
        create_directories(
            [Path(training.root_dir)]
        )
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            training_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_eposhs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_images_size=params.IMAGE_SIZE
        )
        return training_config
    
    def get_evaluation_config(self)-> EvaluationConfig:
        # CORRECTION : Utilisez le nouveau chemin de préparation des données
        eval_config = EvaluationConfig(
            path_of_model=r"artifacts\training\trained_model.h5",
            training_data=r"artifacts\data_preparation\kidney-ct-scan-image",  # Chemin corrigé
            mlflow_uri="https://dagshub.com/Radhwen-HAJRI/Kidney-disease-classification.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config