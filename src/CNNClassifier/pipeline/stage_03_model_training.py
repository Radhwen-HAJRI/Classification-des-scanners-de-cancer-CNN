from src.CNNClassifier.config.configuration import ConfigurationManager
from src.CNNClassifier.components.model_training import Training
from src.CNNClassifier import logging
import shutil
from pathlib import Path


STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_training_config()
        model_trainer = Training(config=model_training_config)
        model_trainer.get_base_model()

        
        root_source = Path("artifacts/data_ingestion/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")
        base_source = root_source / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
        base_target = Path("artifacts/data_ingestion/kidney-ct-scan-image")

        
        base_target.mkdir(parents=True, exist_ok=True)

        
        for cls in ["Normal", "Tumor"]:
            src = base_source / cls
            dst = base_target / cls
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                logging.info(f"Déplacement de {cls} vers {dst}")
                shutil.move(str(src), str(dst))

        logging.info(f"Contenu final de {base_target}: {[p.name for p in base_target.iterdir()]}")

        
        model_trainer.train_valid_generator()

        
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
