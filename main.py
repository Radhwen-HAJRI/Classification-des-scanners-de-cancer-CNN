import sys
import os

# Ajouter src au path pour que Python trouve le package CNNClassifier
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.CNNClassifier.log_config import logging

logging.info("This is a log message from main.py")
print("Log message written successfully!")
