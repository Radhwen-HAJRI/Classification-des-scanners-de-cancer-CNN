import os
from box.exceptions import BoxValueError
import yaml
from src.CNNClassifier.log_config import logging
import json
import joblib
from box import ConfigBox
from ensure import ensure_annotations
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError :
        raise BoxValueError("yaml file is empty")
    except Exception as e:
        raise e 
    


def create_directories(path_to_directories: list) -> None:  # <-- juste list
    from pathlib import Path
    for path in path_to_directories:
        path = Path(path)  # convertit str en Path si nécessaire
        os.makedirs(path, exist_ok=True)
        logging.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    logging.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path, "r") as json_file:
        content = json.load(json_file)
    logging.info(f"json file loaded from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path) -> None:
    joblib.dump(data, path)
    logging.info(f"binary file saved at: {path}")   

@ensure_annotations
def load_bin(path: Path) -> Any:
    data = joblib.load(path)
    logging.info(f"binary file loaded from: {path}")
    return data


def encodeImageIntoBase64(image_path: Path) -> str:
    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode('utf-8')
    return b64_string

def decodeImage(b64_string: str, output_path: Path) -> None:
    img_data = base64.b64decode(b64_string)
    with open(output_path, "wb") as img_file:
        img_file.write(img_data)
    logging.info(f"Image decoded and saved at: {output_path}")

def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"
