import sys
import json
import cv2
from main import SimpleFacerec

def main():
    # Check if the image path is passed as an argument, otherwise, hardcode it for testing
    if len(sys.argv) < 2:
        image_path = r"C:\Users\dell\PycharmProjects\test\pythonProject\images\AhmedHamdy.jpeg"  # Hardcoded image path for testing
        print(f"Using hardcoded image path: {image_path}")
    else:
        image_path = sys.argv[1]  # Image path passed as an argument

    sfr = SimpleFacerec()
    sfr.load_encoding_images("images")

    img = cv2.imread(image_path)
    face_locations, face_names = sfr.detect_known_faces(img)

    results = {
        "face_locations": face_locations.tolist(),
        "face_names": face_names
    }
    print(json.dumps(results))

if __name__ == "__main__":
    main()
