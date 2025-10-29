import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PredictionPipline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # ✅ Chargement du bon modèle
        model_path = os.path.join("artifacts", "training", "trained_model.h5")
        model = load_model(model_path)
        print(f"✅ Modèle chargé depuis : {model_path}")

        # ✅ Prétraitement de l'image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # normalisation
        print("✅ Image prétraitée")

        # ✅ Prédiction
        preds = model.predict(test_image)
        predicted_class = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds)) * 100

        print("✅ Sortie brute du modèle :", preds)
        print("✅ Classe prédite :", predicted_class)
        print("✅ Confiance :", confidence)

        if predicted_class == 1:
            prediction = "Tumor"
        else:
            prediction = "Normal"

        return [{
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%"
        }]
