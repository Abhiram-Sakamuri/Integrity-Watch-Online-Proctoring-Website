import os
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Load the YOLO model
model = YOLO("yolov8m.pt")

# Initialize MediaPipe Face Mesh and Face Detection
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Parameters for detecting cheating
cheating_threshold = 30  # Number of consecutive frames indicating cheating
reset_threshold = 10  # Number of consecutive frames to reset the cheating flag
no_face_threshold = 60  # Number of consecutive frames indicating no face detection
multiple_face_threshold = 60  # Number of consecutive frames indicating multiple face detections

cheating_count = 0
reset_count = 0
no_face_count = 0
multiple_face_count = 0
cheating_flag = False
image_saved_for_cheating = False  # Flag to track whether an image has been saved for the current cheating event

# Create a directory to store cheating detected images if it doesn't exist
output_dir = 'cheating_detected_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize a counter for image filenames
image_counter = 0

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
        mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Object detection using YOLO
        results = model.predict(frame, classes=[67])

        # Perform object detection on the frame
        objects_detected = any(r.boxes.cls.tolist() for r in results)

        if objects_detected:
            # If objects are detected, reset cheating counters
            cheating_count = 0
            reset_count = 0
            no_face_count = 0
            multiple_face_count = 0
            cheating_flag = True
            if not image_saved_for_cheating:  # Save image only if not already saved for the current event
                image_saved_for_cheating = True
                # Save the image where cheating is detected
                image_filename = os.path.join(output_dir, f'cheating_{image_counter}.jpg')
                cv2.imwrite(image_filename, frame)
                image_counter += 1

        # Cheating detection using MediaPipe and OpenCV
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect faces
        face_results = face_detection.process(image)

        # Convert image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if face_results.detections:
            if len(face_results.detections) == 1:
                # Get the bounding box of the detected face
                bboxC = face_results.detections[0].location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                x, y, w, h = bbox

                # Draw the face bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Convert image to RGB and process
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        face_3d = []
                        face_2d = []

                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                                if idx == 1:
                                    nose_2d = (lm.x * iw, lm.y * ih)
                                    nose_3d = (lm.x * iw, lm.y * ih, lm.z * 3000)

                                x, y = int(lm.x * iw), int(lm.y * ih)

                                # Get the 2D Coordinates
                                face_2d.append([x, y])

                                # Get the 3D Coordinates
                                face_3d.append([x, y, lm.z])

                        face_2d = np.array(face_2d, dtype=np.float64)
                        face_3d = np.array(face_3d, dtype=np.float64)

                        # The camera matrix
                        focal_length = 1 * iw
                        cam_matrix = np.array([[focal_length, 0, ih / 2],
                                               [0, focal_length, iw / 2],
                                               [0, 0, 1]])

                        # The distortion parameters
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        # Solve PnP
                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                        # Get rotational matrix
                        rmat, jac = cv2.Rodrigues(rot_vec)

                        # Get angles
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        # Get the y rotation degree
                        x = angles[0] * 360
                        y = angles[1] * 360
                        z = angles[2] * 360

                        # See where the user's head is looking
                        if y < -10:
                            text = "Looking Left"
                            cheating_count += 1
                            reset_count = 0  # Reset the reset counter
                            if cheating_count >= cheating_threshold and not image_saved_for_cheating:
                                cheating_flag = True
                        elif y > 10:
                            text = "Looking Right"
                            cheating_count += 1
                            reset_count = 0  # Reset the reset counter
                            if cheating_count >= cheating_threshold and not image_saved_for_cheating:
                                cheating_flag = True
                        else:
                            text = "Looking Forward"
                            reset_count += 1
                            if reset_count >= reset_threshold:
                                cheating_count = 0  # Reset the cheating counter
                                cheating_flag = False
                                image_saved_for_cheating = False  # Reset the flag

                        # Add the text on the image
                        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            else:
                # Increment multiple face count and check for cheating
                multiple_face_count += 1
                if multiple_face_count >= multiple_face_threshold:
                    cheating_flag = True
                    multiple_face_count = 0
                    # You might want to include additional actions here, such as logging or notifications

                # Display message
                cv2.putText(image, "Multiple faces detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            # Increment no face count and check for cheating
            no_face_count += 1
            if no_face_count >= no_face_threshold:
                cheating_flag = True
                no_face_count = 0
                # You might want to include additional actions here, such as logging or notifications

            # Display message
            cv2.putText(image, "No face detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # If cheating detected and no image has been saved for the current cheating event, save the image
        if cheating_flag and not image_saved_for_cheating:
            # Save the image where cheating is detected
            image_filename = os.path.join(output_dir, f'cheating_{image_counter}.jpg')
            cv2.imwrite(image_filename, image)
            image_counter += 1
            image_saved_for_cheating = True  # Set the flag to indicate that the image has been saved

            cv2.putText(image, "Cheating Detected!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        cv2.imshow("Integrated Detection", image)

        # Check for the 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture object and close all windows
cap.release()
cv2.destroyAllWindows()