# by M.A.G.N.E.U.M
# This code builds a facial emotion recognition model using tensorflow keras and fer2013

import cv2
import numpy as np
from keras.models import model_from_json


def main():
    json_file = open("public/best_model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("public/best_model.h5")
    face_haar_cascade = cv2.CascadeClassifier(
        "database/haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)
    print(cap)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(
            gray_img, 1.1, 6, minSize=(150, 150)
        )
        print(faces_detected)

        for x, y, w, h in faces_detected:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            roi_gray = gray_img[y : y + w, x : x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = np.array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels = img_pixels.astype(np.uint8)
            predictions = model.predict(img_pixels)
            max_index = int(np.argmax(predictions))
            emotions = [
                "neutral",
                "happiness",
                "surprise",
                "sadness",
                "anger",
                "disgust",
                "fear",
            ]
            predicted_emotion = emotions[max_index]
            print(predictions, predicted_emotion)
            cv2.putText(
                img,
                predicted_emotion,
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            resized_img = cv2.resize(img, (1000, 700))
            cv2.imshow("Facial Emotion Recognition", resized_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
