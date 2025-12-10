import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection   #helper function for detecting the face
mp_drawing = mp.solutions.drawing_utils     #helper function for drawing the box

# Initialize face detector (off the shelf pre-trained model)
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face
    results = face_detection.process(rgb_frame)

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

            # Crop face safely
            x1 = max(x1, 0); y1 = max(y1, 0)
            face_crop = frame[y1:y2, x1:x2]

            cv2.imshow("Face Crop", face_crop)

    cv2.imshow("Mediapipe Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
