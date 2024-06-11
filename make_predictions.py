import cv2
import sqlite3
import numpy as np
from keras.models import load_model

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")
faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
signModel = load_model("models/keras_model.h5", compile=False)
class_names = open("models/labels.txt", "r").readlines()

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = (255, 255, 255)
fontWeight = 2
fontBottomMargin = 30
nametagHeight = 50
faceRectangleBorderSize = 2
knownTagColor = (100, 180, 0)
unknownTagColor = (0, 0, 255)
knownFaceRectangleColor = knownTagColor
unknownFaceRectangleColor = unknownTagColor
recognition_count = {}
REQUIRED_RECOGNITION_COUNT = 5
face_recognized = False

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    if not face_recognized:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            customer_uid, confidence = faceRecognizer.predict(gray[y:y + h, x:x + w])
            customer_name = "Unknown"
            nametagColor = unknownTagColor
            faceRectangleColor = unknownFaceRectangleColor

            if 60 < confidence < 85:
                try:
                    conn = sqlite3.connect('customer_faces_data.db')
                    c = conn.cursor()
                    c.execute("SELECT customer_name FROM customers WHERE customer_uid = ?", (customer_uid,))
                    row = c.fetchone()
                except sqlite3.Error as e:
                    print("SQLite error:", e)
                    row = None
                finally:
                    if conn:
                        conn.close()

                if row:
                    customer_name = row[0].split(" ")[0]
                    nametagColor = knownTagColor
                    faceRectangleColor = knownFaceRectangleColor

                    if customer_uid not in recognition_count:
                        recognition_count[customer_uid] = 0
                    recognition_count[customer_uid] += 1

                    if recognition_count[customer_uid] >= REQUIRED_RECOGNITION_COUNT:
                        face_recognized = True
                        current_customer_uid = customer_uid
                        print(f"Face recognized: {customer_name}")
                        break

            cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), faceRectangleColor, faceRectangleBorderSize)
            cv2.rectangle(frame, (x - 22, y - nametagHeight), (x + w + 22, y - 22), nametagColor, -1)
            cv2.putText(frame, f"{customer_name}", (x, y - fontBottomMargin), fontFace, fontScale, fontColor, fontWeight)

    if face_recognized:
        ret, sign_image = camera.read()
        if not ret:
            break

        sign_image_resized = cv2.resize(sign_image, (224, 224), interpolation=cv2.INTER_AREA)
        sign_image_array = np.asarray(sign_image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        sign_image_normalized = (sign_image_array / 127.5) - 1
        prediction = signModel.predict(sign_image_normalized)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        cv2.putText(sign_image, f"Sign: {class_name[2:]} ({confidence_score*100:.2f}%)", (10, 30), fontFace, 1, (0, 255, 0), 2)
        cv2.imshow("Sign Detection", sign_image)

        if class_name[2:].lower() == "class 1" and confidence_score > 0.75:
            try:
                conn = sqlite3.connect('customer_faces_data.db')
                c = conn.cursor()
                c.execute("UPDATE customers SET confirm = 1 WHERE customer_uid = ?", (current_customer_uid,))
                conn.commit()
                print(f"{customer_name} confirmed!")
            except sqlite3.Error as e:
                print("SQLite error:", e)
            finally:
                if conn:
                    conn.close()

            recognition_count = {}
            face_recognized = False

    if not face_recognized:
        cv2.imshow('Detecting Faces...', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
