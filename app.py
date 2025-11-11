from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from transformers import pipeline
from markupsafe import Markup
import pandas as pd
import random
import logging
import os

# === Import du Chatbot (Blueprint) ===
from modules.chatbot.app import chatbot_bp

# === Import du module caméra ===
from modules.livesentiment import gen_frames


# ============================================================
# 🧠 Flask App – Accueil + Sentiment Analysis + Modules IA
# ============================================================
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# === Enregistrement du Blueprint Chatbot ===
app.register_blueprint(chatbot_bp)


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
    """Page d'accueil avec menu principal"""
    return render_template("welcome.html")


# ------------------------------------------------------------
# 🔹 PAGE : Prédire un sentiment
# ------------------------------------------------------------
@app.route("/predictor")
def predictor_page():
    """Affiche la page principale de prédiction de sentiment"""
    return render_template("index.html")


# ------------------------------------------------------------
# 🔹 PAGE : Caméra – Détection d’émotion intégrée
# ------------------------------------------------------------
@app.route("/camera")
def camera_page():
    """Affiche la page caméra avec le design Sentio"""
    return render_template("camera.html")


# ------------------------------------------------------------
# 🔹 Lancement caméra via bouton (JSON)
# ------------------------------------------------------------
@app.route("/run_camera")
def run_camera():
    """Lance la caméra et renvoie un message JSON."""
    return jsonify({"status": "ok", "message": "Caméra lancée avec succès ✅"})


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
# 🔹 API : Prédiction du sentiment + Recommandation automatique
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """Analyse le sentiment d’un texte saisi ou sélectionné et renvoie une recommandation"""
    try:
        data = request.get_json() or {}
        text = data.get("text", "")
        if not text.strip():
            return jsonify({"error": "Aucun texte fourni."}), 400

        # --- Liste de mots à forte polarité ---
        strong_words = [
            "love", "hate", "great", "bad", "amazing", "awful", "horrible", "fantastic",
            "terrible", "happy", "sad", "angry", "wonderful", "disgusting", "excited",
            "boring", "awesome", "worst", "best", "enjoy", "cry", "pain", "fear"
        ]

        tokens = text.lower().split()
        strong_hits = sum(1 for w in tokens if w in strong_words)

        # --- Prédiction DistilBERT ---
        result = sentiment_model(text)[0]
        label = result["label"].upper()
        score = round(float(result["score"]), 3)

        # --- Ajustement intelligent de la neutralité ---
        text_len = len(text.split())
        if score < 0.4:
            label = "NEUTRAL"
        elif 0.4 <= score < 0.65 and strong_hits == 0:
            label = "NEUTRAL"
        elif text_len < 4 and score < 0.7:
            label = "NEUTRAL"
        # Sinon : garder la prédiction du modèle

        # --- Génération d'une recommandation automatique ---
        recommendations = {
            "POSITIVE": [
                "Keep that energy! You’re on the right track 🌟",
                "Stay motivated — good vibes attract good things ✨",
                "Celebrate your wins, big or small 🎉"
            ],
            "NEUTRAL": [
                "Take a deep breath and center yourself 🌿",
                "Maybe take a short walk — it always helps ☕",
                "Keep a balanced mindset, that’s your strength ⚖️"
            ],
            "NEGATIVE": [
                "Tough days don’t last — you’ve got this 💪",
                "Remember: every storm runs out of rain 🌧️☀️",
                "It’s okay to pause — self-care is productive ❤️"
            ]
        }

        recommendation = random.choice(recommendations.get(label, ["Stay calm and move forward."]))

        return jsonify({
            "text": text,
            "sentiment_predicted": label,
            "confidence": score,
            "emotion_strength": strong_hits,
            "recommendation": recommendation
        })

    except Exception as e:
        logging.exception("Erreur de prédiction :")
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# 🔹 API : Tweet aléatoire
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
