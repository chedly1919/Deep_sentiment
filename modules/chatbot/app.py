from flask import Blueprint, render_template, request, jsonify
from joblib import load
from pathlib import Path
from dotenv import load_dotenv
import random
import os

# ============================================================
# 🤖 Chatbot Sentio - Flask Blueprint (intégration principale)
# ============================================================

# Charger les variables d'environnement
load_dotenv()

# --- Déclaration du blueprint ---
chatbot_bp = Blueprint("chatbot", __name__, template_folder="templates")

# --- Chargement du modèle local ---
MODEL_PATH = Path("modules/chatbot/models/sentiment.joblib")
model = load(MODEL_PATH) if MODEL_PATH.exists() else None

conversation_history = []  # Historique local du chat


# ============================================================
# 🔹 Fonction : prédire le sentiment du texte
# ============================================================
def predict_sentiment(text):
    """Utilise le modèle local pour détecter le sentiment."""
    if not model:
        return "neutral", 0.0
    label = model.predict([text])[0]
    proba = model.predict_proba([text])[0].max()
    return label, round(float(proba), 2)


# ============================================================
# 🔹 Page principale du chatbot
# ============================================================
@chatbot_bp.route("/chatbot", methods=["GET"])
def chatbot_page():
    """Affiche la page de discussion du chatbot."""
    return render_template("chat.html")


# ============================================================
# 🔹 API : envoyer un message au chatbot
# ============================================================
@chatbot_bp.route("/chatbot/message", methods=["POST"])
def chatbot_message():
    """Génère une réponse selon le sentiment détecté."""
    data = request.get_json() or {}
    user_text = data.get("message", "").strip()

    if not user_text:
        return jsonify({"response": "Dis-moi quelque chose 😄"}), 400

    sentiment, confidence = predict_sentiment(user_text)

    responses = {
        "positive": [
            "C’est génial 😄 ! Continue sur cette lancée !",
            "Super vibe aujourd’hui 🌟",
            "Tu dégages une belle énergie ✨"
        ],
        "negative": [
            "Oh… je comprends 😔, veux-tu en parler ?",
            "Courage 💪, demain sera meilleur.",
            "Même dans la pluie, il y a un peu de lumière ☀️"
        ],
        "neutral": [
            "Intéressant 🤔, raconte-moi plus !",
            "Je t’écoute attentivement 👂",
            "Dis-m’en un peu plus 😌"
        ]
    }

    # Sélection d'une réponse adaptée
    response = random.choice(responses.get(sentiment, ["Je t’écoute attentivement."]))

    conversation_history.append({
        "user": user_text,
        "bot": response,
        "sentiment": sentiment,
        "confidence": confidence
    })

    return jsonify({
        "response": response,
        "sentiment": sentiment,
        "confidence": confidence
    })


# ============================================================
# 🔹 API : tester uniquement le modèle de sentiment
# ============================================================
@chatbot_bp.route("/chatbot/sentiment", methods=["POST"])
def chatbot_sentiment_test():
    """Permet de tester uniquement la détection de sentiment."""
    data = request.get_json() or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "missing text"}), 400

    label, conf = predict_sentiment(text)
    return jsonify({"sentiment": label, "confidence": conf})


# ============================================================
# 🚀 Exécution autonome (debug local uniquement)
# ============================================================
if __name__ == "__main__":
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(chatbot_bp)
    app.run(host="0.0.0.0", port=5000, debug=True)
