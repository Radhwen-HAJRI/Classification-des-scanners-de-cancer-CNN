from src.CNNClassifier import logging
from src.CNNClassifier.pipeline.stage_01_data_ingetion import DataIngestionTrainingPipeline
from src.CNNClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"
try:
    logging.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logging.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e



STAGE_NAME = "Prepare Base Model"

try:
        logging.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        logging.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e