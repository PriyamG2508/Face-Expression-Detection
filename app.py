import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load model
@st.cache_resource
def load_emotion_model():
    return load_model("emotiondetector.h5")

model = load_emotion_model()
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Preprocess face from frame
def preprocess_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        return roi, (x, y, w, h)
    return None, None

st.title("🧠 Real-Time Emotion Detection")
st.write("Detects emotion from webcam feed using a trained deep learning model.")

# Checkbox to start camera
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

# Initialize camera outside the loop
if run:
    camera = cv2.VideoCapture(0)

    while run:
        success, frame = camera.read()
        if not success:
            st.error("Unable to access webcam.")
            break

        img = cv2.flip(frame, 1)
        face, coords = preprocess_face(img)

        if face is not None:
            pred = model.predict(face)[0]
            label_idx = np.argmax(pred)
            label_text = labels[label_idx]
            x, y, w, h = coords
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)

        FRAME_WINDOW.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Once run becomes False
    camera.release()
    cv2.destroyAllWindows()
else:
    st.warning("Check 'Start Camera' to begin.")
