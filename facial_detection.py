import cv2
import mediapipe as mp

def facial_detection_routine():
    mp_face_detection = mp.solutions.face_detection   #helper function for detecting the face
    mp_drawing = mp.solutions.drawing_utils     #helper function for drawing the box

    # Initialize face detector (off the shelf pre-trained model)
    #model_selection=0 => near-field optimization, like a selfie/webcam feed
    #min_detection_confidence=0.5. error threshold to detect a face. 0->most permissive 1->most strict
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    # Start webcam 
    cap = cv2.VideoCapture(0)

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

                # Crop face safely
                x1 = max(x1, 0); y1 = max(y1, 0)
                face_crop = frame[y1:y2, x1:x2]

                cv2.imshow("Face Crop", face_crop)  #show current annotated window of cropped face

        
        cv2.imshow("Mediapipe Face Detection", frame)   #show current annotated window of frame captured
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   #quit condition

    #release resources
    cap.release()
    cv2.destroyAllWindows()
