from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import face_recognition
import cv2
import os
import threading
import pickle
import numpy as np
import pyodbc

# Update the Flask app to serve static files from the custom images folder
app = Flask(__name__, static_folder="C:/Users/dell/source/repos/PatientSystem/images")
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})

# Preload known faces and data
class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.patient_data = {}


    def load_data_from_database(self, base_url):
        connection = pyodbc.connect(
            'DRIVER={SQL Server};'
            'SERVER=DESKTOP-CLQGA5Q\\SQLEXPRESS;'
            'DATABASE=PatientSystemDB;'
            'Trusted_Connection=yes;'
        )
        cursor = connection.cursor()
        cursor.execute("SELECT Id, Name, Dob, Mobileno, Nationalno, FaceImg FROM dbo.Patients")
        patients = cursor.fetchall()

        for patient in patients:
            patient_id, name, dob, mobile_no, national_no, face_img = patient
            image_path = os.path.join("images", str(face_img))  # Assuming the images are in 'static/images'
            if not os.path.exists(image_path):
                continue

            # Process image
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                encoding = face_recognition.face_encodings(rgb_image)[0]
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)

                # Generate full URL for the face image
                face_img_url = f"{base_url}/images/{face_img}"

                self.patient_data[name] = {
                    "id": patient_id,
                    "dob": dob,
                    "mobileno": mobile_no,
                    "nationalno": national_no,
                    "faceImg": face_img_url,  # Use full URL for the image
                }
            except IndexError:
                print(f"Face not found in image for patient {name}. Skipping...")

        connection.close()

    def save_encodings_to_file(self):
        with open("face_encodings.pkl", "wb") as f:
            pickle.dump((self.known_face_encodings, self.known_face_names, self.patient_data), f)

    def load_encodings_from_file(self, base_url):
        if os.path.exists("face_encodings.pkl"):
            with open("face_encodings.pkl", "rb") as f:
                self.known_face_encodings, self.known_face_names, self.patient_data = pickle.load(f)

    def recognize_face(self, unknown_image):
        unknown_encoding = face_recognition.face_encodings(unknown_image)
        if not unknown_encoding:
            return None, None

        unknown_encoding = unknown_encoding[0]
        matches = face_recognition.compare_faces(self.known_face_encodings, unknown_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(self.known_face_encodings, unknown_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            matched_name = self.known_face_names[best_match_index]
            return matched_name, self.patient_data.get(matched_name)

        return None, None


facerec = SimpleFacerec()
facerec.load_encodings_from_file("http://localhost:5000")  # Assuming base URL is localhost:5000


@app.route('/images/<filename>')
def serve_images(filename):
    # Now Flask will serve images from the 'images' folder directly
    return send_from_directory(app.static_folder, filename)


@app.route('/detectAndFind', methods=['POST'])
def detect_and_find():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = os.path.join("uploads", file.filename)
    file.save(filename)

    def process_image(base_url):
        unknown_image = face_recognition.load_image_file(filename)
        matched_name, patient_data = facerec.recognize_face(unknown_image)

        nonlocal result
        if matched_name:
            result = {"isMatch": True, "patientName": matched_name, "patientData": patient_data}
        else:
            result = {"isMatch": False, "patientName": "Unknown", "patientData": {}}

        # Log the generated faceImg URL and patient data
        print(f"Patient data: {patient_data}")

    result = None
    base_url = f"{request.scheme}://{request.host}"  # Generate the base URL dynamically
    thread = threading.Thread(target=process_image, args=(base_url,))
    thread.start()
    thread.join()

    return jsonify(result)


if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
