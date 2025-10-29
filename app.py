from flask import Flask, jsonify, render_template, request
import os
import traceback
from flask_cors import CORS, cross_origin
from CNNClassifier.utils.common import decodeImage
from CNNClassifier.pipeline.prediction import PredictionPipline

# Configuration d'encodage
os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)

# -------- Classe principale --------
class ClientApp:
    def __init__(self, app):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipline(self.filename)

# Crée une seule instance globale
clApp = ClientApp(app)

# -------- Routes Flask --------
@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():
    os.system("dvc repro")
    return "Training done successfully"

@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, filename=clApp.filename)
        print("✅ Image décodée :", clApp.filename, os.path.exists(clApp.filename))
        
        result = clApp.classifier.predict()
        print("✅ Résultat brut :", result)

        # ⚙️ Normalise la réponse au bon format
        if isinstance(result, dict):
            response = [result]
        elif isinstance(result, (list, tuple)):
            response = result
        else:
            response = [{"prediction": str(result)}]

        return jsonify(response)

    except Exception as e:
        print("\n❌ ERREUR lors de la prédiction :")
        traceback.print_exc()  # affiche l'erreur complète dans le terminal
        return jsonify({"error": str(e)}), 500

# -------- Lancement --------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
