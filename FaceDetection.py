import cv2
import numpy as np
from keras.models import model_from_json

model = model_from_json(open("fer.json", "r").read())
model.load_weights("fer.h5")

# 防止载入不必要的数据
cv2.ocl.setUseOpenCL(False)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
emotion_list = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Sad", 5: "Happy", 6: "Neutral"}

while True:

    ret, frames = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.32,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 7)
        roi_gray = gray[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        max_index = int(np.argmax(prediction))
        cv2.putText(frames, emotion_list[max_index], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', cv2.resize(frames, (1000, 700), interpolation=cv2.INTER_CUBIC))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
