import cv2
import sqlite3

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")
faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = (255, 255, 255)
fontWeight = 2
fontBottomMargin = 30

nametagColor = (100, 180, 0)
nametagHeight = 50

faceRectangleBorderColor = nametagColor
faceRectangleBorderSize = 2

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:

        customer_uid, Confidence = faceRecognizer.predict(gray[y:y + h, x:x + w])
        try:
            conn = sqlite3.connect('customer_faces_data.db')
            c = conn.cursor()
        except sqlite3.Error as e:
            print("SQLite error:", e)


        c.execute("SELECT customer_name FROM customers WHERE customer_uid LIKE ?", (f"{customer_uid}%",))
        row = c.fetchone()
        if row:
            customer_name = row[0].split(" ")[0]
        else:
            customer_name = "Unknown"

                    
        if 45<Confidence<100:
            cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), faceRectangleBorderColor, faceRectangleBorderSize)

            cv2.rectangle(frame, (x - 22, y - nametagHeight), (x + w + 22, y - 22), nametagColor, -1)
            cv2.putText(frame, str(customer_name) + ": " + str(round(Confidence, 2)) + "%", (x, y-fontBottomMargin), fontFace, fontScale, fontColor, fontWeight)

    cv2.imshow('Detecting Faces...', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
conn.close()
