import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from src.CNNClassifier.entity.config_entity import EvaluationConfig
from src.CNNClassifier.utils.common import save_json, load_json, read_yaml
import dagshub
import os
class Evaluation:
    def __init__(self, config:EvaluationConfig):
        self.config = config
        
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale= 1./255,
            validation_split = 0.30
        )
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        valid_datagenerator=tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self._valid_generator=valid_datagenerator.flow_from_directory(
            directory= self.config.training_data,
            subset= "validation",
            shuffle=False,
            class_mode="sparse",
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path)->tf.keras.Model:
        return tf.keras.models.load_model(path)
    def Evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self._valid_generator)
        
    def save_score(self):
        score = {
            "loss":self.score[0],
            "accuracy":self.score[1]
        }
        save_json(path=Path("score.json"),data=score)
    def log_into_mlflow(self):
        os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/Radhwen-HAJRI/Kidney-disease-classification.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"]="Radhwen-HAJRI"
        os.environ["MLFLOW_TRACKING_PASSWORD"]="78e5e078c134d008091b62c2d4e5c913bba61b00"
        
        dagshub.init(repo_owner='Radhwen-HAJRI', repo_name='Kidney-disease-classification', mlflow=True)
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("loss",self.score[0])
            mlflow.log_metric("accuracy",self.score[1])
            
            mlflow.keras.log_model(
            self.model,
            "model",
            registered_model_name="VGG16_Kidney_disease_classification"
                )
                