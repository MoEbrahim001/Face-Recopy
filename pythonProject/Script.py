from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import face_recognition
import cv2
import os
import numpy as np
import pyodbc

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.55
        self.image_folder = r'C:\Users\dell\source\repos\PatientSystem\images'
        self.patient_data = {}

    def connect_to_database(self):
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

    def load_encoding_images(self, base_url):
        connection = self.connect_to_database()
        if connection is None:
            return

        cursor = connection.cursor()
        cursor.execute("SELECT Id, Name, Dob, Mobileno, Nationalno, FaceImg FROM dbo.Patients")
        patients = cursor.fetchall()

        for patient in patients:
            patient_id, name, dob, mobile_no, national_no, face_img = patient
            image_file = os.path.join(self.image_folder, str(face_img))

            if not os.path.exists(image_file):
                continue

            img = cv2.imread(image_file)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                img_encoding = face_recognition.face_encodings(rgb_img)[0]
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(name)

                face_img_url = f"{base_url}/images/{face_img}"
                self.patient_data[name] = {
                    "mobileno": mobile_no,
                    "nationalno": national_no,
                    "id": patient_id,
                    "dob": dob,
                    "faceImg": face_img_url
                }
            except Exception as e:
                print(f"Error processing {name}: {e}")

        connection.close()

    def compare_faces(self, unknown_image):
        unknown_encoding = face_recognition.face_encodings(unknown_image)
        if not unknown_encoding:
            return None

        unknown_encoding = unknown_encoding[0]
        matches = face_recognition.compare_faces(self.known_face_encodings, unknown_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(self.known_face_encodings, unknown_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            return self.known_face_names[best_match_index]
        return None

@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('C:/Users/dell/source/repos/PatientSystem/images', filename)

@app.route('/detectAndFind', methods=['POST'])
def detect_and_find():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    unknown_image = face_recognition.load_image_file(file_path)
    facerec = SimpleFacerec()
    base_url = f"{request.scheme}://{request.host}"
    facerec.load_encoding_images(base_url)

    matched_name = facerec.compare_faces(unknown_image)
    if matched_name:
        patient_data = facerec.patient_data.get(matched_name)
        return jsonify({"isMatch": True, "patientName": matched_name, "patientData": patient_data})

    return jsonify({"isMatch": False, "patientName": "Unknown", "patientData": {}})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
