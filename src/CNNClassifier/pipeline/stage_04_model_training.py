from src.CNNClassifier.config.configuration import ConfigurationManager
from src.CNNClassifier.components.model_training import Training
from src.CNNClassifier import logging

STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_training_config()
        model_trainer = Training(config=model_training_config)
        model_trainer.get_base_model()
        
        # Génération des données d'entraînement et validation
        model_trainer.train_valid_generator()
        
        # Démarrage de l'entraînement
        model_trainer.train()


if __name__ == "__main__":
    try:
        logging.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logging.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e