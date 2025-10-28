from src.CNNClassifier.config.configuration import ConfigurationManager
from src.CNNClassifier import logging
import shutil
from pathlib import Path

STAGE_NAME = "Data Preparation Stage"

class DataPreparationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        # Configuration des chemins
        root_source = Path("artifacts/data_ingestion/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")
        base_source = root_source / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
        base_target = Path("artifacts/data_preparation/kidney-ct-scan-image")
        
        # Création du dossier cible
        base_target.mkdir(parents=True, exist_ok=True)
        
        # Copie des données pour l'entraînement (seulement Normal et Tumor)
        for cls in ["Normal", "Tumor"]:
            src = base_source / cls
            dst = base_target / cls
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)  # Nettoyer si existe déjà
                logging.info(f"Copying {cls} to {dst}")
                shutil.copytree(str(src), str(dst))  # Copie récursive
        
        logging.info(f"Final content of {base_target}: {[p.name for p in base_target.iterdir()]}")
        logging.info("Data preparation completed successfully")


if __name__ == "__main__":
    try:
        logging.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataPreparationPipeline()
        obj.main()
        logging.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e