from src.CNNClassifier.config.configuration import ConfigurationManager
from CNNClassifier.components.model_evaluation_mlflow import Evaluation
from src.CNNClassifier import logging

STAGE_NAME = "Model Evaluation"

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(config=eval_config)
        evaluation.Evaluation()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logging.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logging.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e