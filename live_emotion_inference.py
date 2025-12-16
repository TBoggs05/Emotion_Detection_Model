import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
#import our model we built in emotion_detection_model.py
#emotion_model = tf.keras.models.load_model('models/emotion_model_50_percent.keras')
emotion_model = tf.keras.models.load_model('models/emotion_model_fine_tuned_v2.keras')
# Show the model architecture
emotion_model.summary()
T = np.load("models/temperature.npy")

emotion_labels = ["angry","fear","happy","sad","surprise","neutral"]
sad_idx = emotion_labels.index("sad")
neutral_idx = emotion_labels.index("neutral")
happy_idx = emotion_labels.index("happy")
pred_buffer = deque(maxlen=10)

mp_face_detection = mp.solutions.face_detection   #helper function for detecting the face
mp_drawing = mp.solutions.drawing_utils     #helper function for drawing the box

    # Initialize face detector (off the shelf pre-trained model)
    #model_selection=0 => near-field optimization, like a selfie/webcam feed
    #min_detection_confidence=0.5. error threshold to detect a face. 0->most permissive 1->most strict
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    # Start webcam 
cap = cv2.VideoCapture(0)

current_emotion = ""
while True:
        ret, frame = cap.read() #ret->frame returned bool | frame->frame object returned from cap.read
        if not ret: #break if feed dies
            break

        # Convert to RGB for Mediapipe (OpenCV gives BGR for some reason)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face
        results = face_detection.process(rgb_frame) #results.detections is a list of faces found, with location info

        if results.detections:  #if the model determines a face feature
            for detection in results.detections:    
                #draw the face onto the screen
                mp_drawing.draw_detection(frame, detection)
                #extracting the face by 
                h, w, _ = frame.shape
                bbox = detection.location_data.relative_bounding_box
                #get coordinates of opencv box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                face_x_padding = int(0.1 * (x2 - x1))
                face_y_padding = int(0.1 * (y2 - y1))
                # Crop face safely
                x1 = max(x1, 0) - face_x_padding; y1 = max(y1, 0) - face_y_padding
                y2 = y2+face_y_padding
                x2= x2+face_x_padding
                face_crop = rgb_frame[y1:y2, x1:x2]
                #face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                try:
                    face_img = cv2.resize(face_crop, (96, 96))
                except:
                     continue
                #normalize + expand dims on face
                #face_img = face_img / 255.0
                face_img = face_img.astype(np.float32)
                #face_img = preprocess_input(face_img)
                face_img = np.expand_dims(face_img, axis=0)
                #pred = emotion_model.predict(face_img)
                #print(pred)
                #temporal smoothing
                #pred_buffer.append(pred[0])
                #avg_pred = np.mean(pred_buffer, axis=0)
                #emotion = emotion_labels[np.argmax(avg_pred)]
                #SWAP TEMPORAL SMOOTHING LOGIC FOR THIS ON EXPIRMENTAL MODEL
                logits = emotion_model.predict(face_img, verbose=0)     # shape (1, C)
                probs  = tf.nn.softmax(logits / T).numpy()[0]          # calibrated probs
                pred_buffer.append(probs)
                avg_pred = np.mean(pred_buffer, axis=0)
                emotion = emotion_labels[int(np.argmax(avg_pred))]

                '''
                if (
                    avg_pred[sad_idx] > 0.20 and
                    avg_pred[happy_idx] < 0.30 and
                    avg_pred[sad_idx] > avg_pred[neutral_idx] * 0.85
                ):
                    emotion = "sad"
                if (
                    emotion == "neutral" and
                    avg_pred[sad_idx] > 0.18 and
                    avg_pred[happy_idx] < 0.25
                ):
                    emotion = "sad"
                top = np.max(avg_pred)
                second = np.partition(avg_pred, -2)[-2]

                if top - second < 0.08:
                    emotion = "neutral"
                    '''
                if(current_emotion != emotion):
                    current_emotion = emotion
                    #pred_buffer.clear()
                    print('new emtion:'+emotion)
                cv2.putText(frame, emotion, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.imshow("Face Crop", face_crop)  #show current annotated window of cropped face

        
        cv2.imshow("Mediapipe Face Detection", frame)   #show current annotated window of frame captured
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   #quit condition


#release resources
cap.release()
cv2.destroyAllWindows()

