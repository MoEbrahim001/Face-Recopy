import face_recognition
import cv2
import os
import numpy as np
import pyodbc
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Import CORS for enabling cross-origin requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})
  # Allow requests from Angular frontend

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.55
        self.image_folder = r'C:\Users\dell\source\repos\PatientSystem\images'  # Folder where patient images are stored

    def connect_to_database(self):
        """
        Establish connection to the SQL Server database.
        """
        try:
            connection = pyodbc.connect(
                'DRIVER={SQL Server};'
                'SERVER=DESKTOP-CLQGA5Q\\SQLEXPRESS;'
                'DATABASE=PatientSystemDB;'
                'Trusted_Connection=yes;'
            )
            return connection
        except Exception as e:
            print(f"Database connection error: {e}")
            return None

    def load_encoding_images(self):
        """
        Load encoding images from the database and images folder.
        """
        # Load from the database
        connection = self.connect_to_database()
        if connection is None:
            print("Failed to connect to database.")
            return

        cursor = connection.cursor()
        cursor.execute("SELECT FaceImg, Name FROM dbo.Patients")  # Fetch FaceImg and Name
        patients = cursor.fetchall()

        if not patients:
            print("No patient data found in the database.")
            return

        print(f"{len(patients)} patients found in the database.")

        # Load images and their encodings
        for patient in patients:
            face_img, name = patient  # Extract FaceImg and Name
            image_file = os.path.join(self.image_folder, face_img)  # Use FaceImg as the filename

            if not os.path.exists(image_file):
                print(f"Image file not found: {image_file} for patient {name}")
                continue

            img = cv2.imread(image_file)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                # Get encoding
                img_encoding = face_recognition.face_encodings(rgb_img)[0]
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(name)
                print(f"Loaded encoding for {name}")
            except Exception as e:
                print(f"Error processing {name}: {e}")

        connection.close()
        print("Encoding images loaded successfully.")

    def compare_faces(self, unknown_image):
        """
        Compare uploaded image against known face encodings.
        """
        # Extract face encoding from the uploaded image
        unknown_encoding = face_recognition.face_encodings(unknown_image)
        if not unknown_encoding:
            return None  # No faces found

        unknown_encoding = unknown_encoding[0]

        # Compare the uploaded image with known encodings
        matches = face_recognition.compare_faces(self.known_face_encodings, unknown_encoding,tolerance=0.5)
        face_distances = face_recognition.face_distance(self.known_face_encodings, unknown_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            return self.known_face_names[best_match_index]
        return None


@app.route('/detectAndFind', methods=['POST'])
def detect_and_find():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file temporarily
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)  # Ensure this path exists
    file.save(file_path)

    # Load the uploaded image
    unknown_image = face_recognition.load_image_file(file_path)

    # Initialize SimpleFacerec class and load encodings from the database
    facerec = SimpleFacerec()
    facerec.load_encoding_images()

    # Compare the uploaded image with known encodings
    matched_name = facerec.compare_faces(unknown_image)

    if matched_name:
        return jsonify({"isMatch": True, "patientName": matched_name})
    else:
        return jsonify({"isMatch": False, "patientName": "Unknown"})


if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Run the Flask app
    app.run(debug=True)
