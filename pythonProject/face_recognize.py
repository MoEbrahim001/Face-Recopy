import cv2
from main import SimpleFacerec

# Encode faces from a folder and database
sfr = SimpleFacerec()
sfr.load_encoding_images(r"C:\Users\dell\source\repos\PatientSystem\images")  # Specify the folder path containing patient images

# Load Camera
cap = cv2.VideoCapture(0)  # Start with index 0

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Failed to capture frame. Exiting.")
        break

    # Detect Faces
    try:
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
    except Exception as e:
        print(f"Detection error: {e}")

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
