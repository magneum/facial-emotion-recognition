# by M.A.G.N.E.U.M
# This code builds a facial emotion recognition model using tensorflow keras and fer2013

import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model("public/model/FER_model.h5")
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
color_map = {
    "Angry": (0, 0, 255),
    "Disgust": (255, 0, 0),
    "Fear": (128, 0, 128),
    "Happy": (0, 255, 0),
    "Sad": (0, 0, 128),
    "Surprise": (255, 255, 0),
    "Neutral": (128, 128, 128),
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "public/src/haarcascade_frontalface_default.xml"
)


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    reshaped = resized.reshape(1, 48, 48, 1)
    normalized = reshaped.astype("float32") / 255.0
    return normalized


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        face_image = frame[y : y + h, x : x + w]
        preprocessed = preprocess_image(face_image)
        prediction = model.predict(preprocessed)
        predicted_emotion = np.argmax(prediction)
        emotion_label = emotion_labels[predicted_emotion]
        prediction_rate = prediction[0][predicted_emotion]
        color = color_map[emotion_label]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            emotion_label + f" ({prediction_rate:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
