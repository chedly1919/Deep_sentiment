from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from transformers import pipeline
from markupsafe import Markup
import pandas as pd
import random
import logging
import os

# === Import du module caméra ===
from modules.livesentiment import gen_frames

# ============================================================
# 🧠 Flask App – Accueil + Sentiment Analysis + Modules IA
# ============================================================

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ------------------------------------------------------------
# 🔹 Chargement du modèle Transformers
# ------------------------------------------------------------
print("🔄 Chargement du modèle DistilBERT...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("✅ Modèle chargé avec succès !")

# ------------------------------------------------------------
# 🔹 Chargement du dataset nettoyé (Sentiment140 clean)
# ------------------------------------------------------------
csv_path = "data/training_clean.csv"

if not os.path.exists(csv_path):
    print(f"⚠️ Fichier CSV introuvable à : {csv_path}")
    df = pd.DataFrame(columns=["sentiment", "text"])
else:
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"✅ Dataset nettoyé chargé : {len(df)} tweets")


# ------------------------------------------------------------
# 🔹 PAGE D’ACCUEIL (Welcome)
# ------------------------------------------------------------
@app.route("/")
def welcome():
    """Page d'accueil avec menu"""
    return render_template("welcome.html")


# ------------------------------------------------------------
# 🔹 PAGE : Prédire un sentiment
# ------------------------------------------------------------
@app.route("/predictor")
def predictor_page():
    """Affiche la page principale de prédiction de sentiment avec bouton retour"""
    back_button_html = """
    <div style='position:absolute; top:20px; left:20px;'>
        <button onclick="window.location.href='/'" 
            style='background:#007bff;color:white;border:none;
            padding:10px 15px;border-radius:8px;font-size:14px;
            cursor:pointer;transition:0.3s;'>⬅️ Retour à l'accueil</button>
    </div>
    """
    page_content = render_template("index.html")
    return Markup(back_button_html + page_content)


# ------------------------------------------------------------
# 🔹 PAGE : Chatbot émotionnel (placeholder)
# ------------------------------------------------------------
@app.route("/chatbot")
def chatbot_page():
    return """
    <div style='text-align:center; font-family:sans-serif; margin-top:40px;'>
        <div style='position:absolute; top:20px; left:20px;'>
            <button onclick="window.location.href='/'" 
                style='background:#007bff;color:white;border:none;
                padding:10px 15px;border-radius:8px;font-size:14px;cursor:pointer;'>⬅️ Retour à l'accueil</button>
        </div>
        <h1>🤖 Chatbot émotionnel – en développement...</h1>
    </div>
    """


# ------------------------------------------------------------
# 🔹 PAGE : Système de recommandation (placeholder)
# ------------------------------------------------------------
@app.route("/recommend")
def recommend_page():
    return """
    <div style='text-align:center; font-family:sans-serif; margin-top:40px;'>
        <div style='position:absolute; top:20px; left:20px;'>
            <button onclick="window.location.href='/'" 
                style='background:#007bff;color:white;border:none;
                padding:10px 15px;border-radius:8px;font-size:14px;cursor:pointer;'>⬅️ Retour à l'accueil</button>
        </div>
        <h1>🎯 Système de recommandation – en développement...</h1>
    </div>
    """


# ------------------------------------------------------------
# 🔹 PAGE : Caméra – Détection d’émotion intégrée
# ------------------------------------------------------------
@app.route("/camera")
def camera_page():
    """Affiche la page caméra intégrée directement dans Flask"""
    return """
    <div style='text-align:center; font-family:sans-serif; margin-top:30px;'>
        <div style='position:absolute; top:20px; left:20px;'>
            <button onclick="window.location.href='/'"
                style='background:#007bff;color:white;border:none;
                padding:10px 15px;border-radius:8px;font-size:14px;cursor:pointer;'>⬅️ Retour à l'accueil</button>
        </div>
        <h1>🎥 Détection d’émotion en direct</h1>
        <p>Analyse de ton humeur via la webcam en temps réel :</p>
        <div style='display:flex;justify-content:center;margin-top:20px;'>
            <img src="/video_feed" width="700" style="border-radius:10px;box-shadow:0 0 10px rgba(0,0,0,0.3);"/>
        </div>
    </div>
    """


# ------------------------------------------------------------
# 🔹 FLUX VIDÉO – Diffusion en continu (MJPEG)
# ------------------------------------------------------------
@app.route("/video_feed")
def video_feed():
    """Diffuse le flux vidéo depuis la webcam"""
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ------------------------------------------------------------
# 🔹 API : Liste déroulante des tweets
# ------------------------------------------------------------
@app.route("/tweets", methods=["GET"])
def get_all_tweets():
    """Retourne un échantillon de tweets pour la liste déroulante"""
    try:
        if df.empty:
            return jsonify({"error": "Le dataset est vide ou introuvable."}), 404

        sample_df = df.sample(min(1000, len(df)))
        tweets = [{"text": t} for t in sample_df["text"].tolist()]
        return jsonify({"tweets": tweets})

    except Exception as e:
        logging.exception("Erreur récupération tweets :")
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# 🔹 API : Prédiction du sentiment avec DistilBERT
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """Analyse le sentiment d’un texte saisi ou sélectionné"""
    try:
        data = request.get_json() or {}
        text = data.get("text", "")
        if not text.strip():
            return jsonify({"error": "Aucun texte fourni."}), 400

        result = sentiment_model(text)[0]
        label = result["label"]
        score = round(float(result["score"]), 3)

        return jsonify({
            "text": text,
            "sentiment_predicted": label,
            "confidence": score
        })

    except Exception as e:
        logging.exception("Erreur de prédiction :")
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# 🔹 API : Tweet aléatoire (optionnel)
# ------------------------------------------------------------
@app.route("/tweet", methods=["GET"])
def get_random_tweet():
    """Retourne un tweet aléatoire du dataset"""
    try:
        if df.empty:
            return jsonify({"error": "Le dataset est vide ou introuvable."}), 404

        row = df.sample(1).iloc[0]
        sentiment_label = "POSITIVE" if row["sentiment"] == 1 else "NEGATIVE"

        return jsonify({
            "text": row["text"],
            "sentiment_original": sentiment_label
        })

    except Exception as e:
        logging.exception("Erreur lecture CSV :")
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# 🔹 Lancement du serveur Flask
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
