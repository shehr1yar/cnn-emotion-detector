import os
import cv2
from keras.models import model_from_json
import numpy as np

MODEL_DIR = os.path.join("models", "model1_accuracy_66.5")

json_path = os.path.join(MODEL_DIR, "emotion_detector_model_1.json")
weights_path = os.path.join(MODEL_DIR, "emotion_detector_model_1.h5")

with open(json_path, "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights(weights_path)

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
meme_images = {}
MEME_SIZE = (240, 240)  # meme size

for label_name in labels.values():
    meme_path = os.path.join("memes", f"{label_name}.jpg")
    img = cv2.imread(meme_path)
    if img is not None:
        meme_images[label_name] = cv2.resize(img, MEME_SIZE)
    else:
        print(f"Warning: Could not load meme image at {meme_path}")

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)

while True:
    i, im = webcam.read()
    if not i:
        break

    # Flip horizontally
    im = cv2.flip(im, 1)

    im = cv2.resize(im, (700, 500))

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Scale face detection down for fast processing
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
    
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            print("Predicted Output:", prediction_label)

            cv2.putText(
                im, 
                str(prediction_label).capitalize(), 
                (p, max(q - 15, 30)), 
                cv2.FONT_HERSHEY_DUPLEX, 
                1.0, 
                (0, 255, 0), 
                2, 
                cv2.LINE_AA
            )

            # Overlay Meme
            if prediction_label in meme_images:
                meme = meme_images[prediction_label]
                mh, mw, _ = meme.shape
                
                # Position overlay: 40px padding from the right edge, vertically centered
                h, w, _ = im.shape
                y_offset = (h - mh) // 2 + 50  # lower middle region
                x_offset = w - mw - 40         # right edge with margin
                
                im[y_offset : y_offset + mh, x_offset : x_offset + mw] = meme

        cv2.imshow("Emotion Recognizer", im)
        
        # loop break "esc" key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    except cv2.error:
        pass

webcam.release()
cv2.destroyAllWindows()