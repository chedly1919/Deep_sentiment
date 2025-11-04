import cv2
from fer import FER
import numpy as np
import random
import webbrowser
from collections import deque
import atexit

# ============================================================
# 🎥 MODULE : Détection d'émotions (compatible Flask + autonome)
# ============================================================

# --- Initialisation du détecteur FER ---
detector = FER(mtcnn=True)
camera = cv2.VideoCapture(0)
history = deque(maxlen=10)

# --- Messages selon l’humeur ---
positive_messages = [
    "Super ! Continue à sourire 😄",
    "Tu rayonnes aujourd’hui 🌟",
    "Félicitations ! Garde cette bonne énergie 💪",
    "Défi du jour : fais sourire quelqu’un autour de toi 😁"
]

neutral_messages = [
    "On dirait que tu es calme... un petit sourire ? 😊",
    "Relax, tout va bien ✨",
    "Petit conseil : pense à un bon souvenir 😌",
    "Et si tu mettais ta musique préférée ? 🎶"
]

negative_quotes = [
    "Ne te décourage pas, les nuages passent toujours ☀️",
    "Chaque jour est une nouvelle chance 💫",
    "Respire, souris, recommence 🌿",
    "Tu es plus fort(e) que tu ne le penses 💪"
]

relaxing_songs = [
    "https://www.youtube.com/watch?v=2OEL4P1Rz04",  # Chill music
    "https://www.youtube.com/watch?v=1ZYbU82GVz4",  # Relaxing piano
    "https://www.youtube.com/watch?v=DWcJFNfaw9c"   # Calm background
]


# --- Nettoyage automatique à la fermeture ---
@atexit.register
def release_camera():
    """Ferme la webcam proprement si Flask s'arrête"""
    if camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()


# ============================================================
# 💬 Réactions émotionnelles (messages + musique)
# ============================================================
def react_to_emotion(emotion):
    """Affiche un message et ouvre une musique selon l'émotion"""
    if emotion == "happy":
        print("🎉", random.choice(positive_messages))

    elif emotion == "sad":
        print("💬", random.choice(negative_quotes))
        song = random.choice(relaxing_songs)
        print(f"🎵 Musique relaxante : {song}")
        webbrowser.open(song)  # ✅ ouvre dans un nouvel onglet navigateur

    elif emotion == "neutral":
        print("🙂", random.choice(neutral_messages))


# ============================================================
# 🧠 Fonction de génération des frames (pour Flask)
# ============================================================
last_emotion = None
last_detected_emotion = None

def gen_frames():
    """Génère les frames pour diffusion MJPEG dans Flask"""
    global last_emotion, last_detected_emotion

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = detector.detect_emotions(frame)
        if results:
            face = results[0]
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]

            # Lisser les émotions
            history.append(emotions)
            avg_emotions = {k: np.mean([h[k] for h in history]) for k in emotions}
            dominant = max(avg_emotions, key=avg_emotions.get)
            last_detected_emotion = dominant

            # 🎵 Réagir si l’émotion change
            if dominant != last_emotion:
                react_to_emotion(dominant)
                last_emotion = dominant

            # Couleur selon émotion
            color = (0, 255, 0) if dominant == "happy" else \
                    (0, 0, 255) if dominant == "sad" else (200, 200, 200)

            # Dessiner cadre + label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 36), (x + w, y), color, -1)
            cv2.putText(frame, dominant.upper(), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Encodage JPEG pour Flask
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ============================================================
# 🌐 Route Flask : renvoyer la dernière émotion
# ============================================================
def get_last_emotion():
    """Retourne la dernière émotion détectée"""
    global last_detected_emotion
    return last_detected_emotion or "neutral"


# ============================================================
# 🚀 Mode autonome (si lancé directement)
# ============================================================
def start_emotion_detection():
    """Lance la détection d’émotions localement (hors Flask)"""
    print("🎥 Assistant émotionnel – appuie sur 'q' pour quitter.")
    last_emotion_local = None

    while True:
        ret, frame = camera.read()
        if not ret:
            print("⚠️ Erreur : impossible d'accéder à la caméra.")
            break

        results = detector.detect_emotions(frame)
        if results:
            face = results[0]
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]

            history.append(emotions)
            avg_emotions = {k: np.mean([h[k] for h in history]) for k in emotions}
            dominant = max(avg_emotions, key=avg_emotions.get)

            if dominant != last_emotion_local:
                react_to_emotion(dominant)
                last_emotion_local = dominant

            color = (0, 255, 0) if dominant == "happy" else \
                    (0, 0, 255) if dominant == "sad" else (200, 200, 200)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, dominant.upper(), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Assistant Émotionnel (Appuie sur 'q' pour quitter)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_emotion_detection()
