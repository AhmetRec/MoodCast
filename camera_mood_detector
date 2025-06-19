import cv2
from deepface import DeepFace

# Duyguyu MoodCast formatına eşleyelim
emotion_map = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "disgust": "angry",
    "fear": "anxious",
    "surprise": "calm",
    "neutral": "neutral"
}

def get_camera_mood():
    cap = cv2.VideoCapture(0)
    detected_mood = "neutral"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            print(f"Algılanan duygu: {emotion}")
            mapped = emotion_map.get(emotion, "neutral")

            # Neutral değilse hemen return et ve döngüyü kır
            if mapped != "neutral":
                detected_mood = mapped
                break

        except Exception as e:
            print(f"Hata: {e}")
            pass

        # Görüntü göster (opsiyonel)
        cv2.imshow("Kameradan Mood Tespiti", frame)

        # Elle çıkmak için Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return detected_mood
