from datetime import datetime, date
import cv2
import base64
import numpy as np
import face_recognition
import os
import pickle
import re
from flask import Flask, render_template, request, redirect, url_for, session
import firebase_admin
from firebase_admin import credentials, db, storage
import cvzone

# Initialize Firebase app and other configurations
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': FIREBASE_DATABASE_URL,
    'storageBucket': FIREBASE_STORAGE_BUCKET_URL
})

app = Flask(__name__)
app.secret_key = b'SECRET_KEY'
datetoday2 = date.today().strftime("%d-%B-%Y")
bucket = storage.bucket()

# Function to update attendance in Firebase
def update_attendance(student_id):
    ref_students = db.reference("Students")
    ref_attendance = db.reference("Attendance")

    # Get student details from database
    student_info = ref_students.child(student_id).get()
    if student_info:
        current_date = date.today().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")

        # Construct attendance data with student details
        data = {
            'name': student_info.get('name'),
            'major': student_info.get('major'),
            'date': current_date,
            'time': current_time
        }

        # Push attendance data to Firebase with student_id as key
        ref_attendance.child(student_id).set(data)
    else:
        print(f"Student with ID {student_id} not found in database.")

def extract_attendance():
    ref = db.reference('Attendance')
    snapshot = ref.get()
    if snapshot:
        names = []
        rolls = []
        times = []
        majors = []  # Added to store majors
        for key, value in snapshot.items():
            names.append(value.get('name'))
            rolls.append(key)  # Assuming 'Roll' is the key itself
            times.append(value.get('time'))
            majors.append(value.get('major'))
        l = len(names)
        return names, rolls, times, majors, l
    else:
        return [], [], [], [], 0



# Function to retrieve total registered users from Firebase
def totalreg():
    ref = db.reference('Students')
    snapshot = ref.get()

    if snapshot:
        return len(snapshot)  # Return the number of items in the snapshot (number of users)
    else:
        return 0  # Return 0 if no users are found in the snapshot

# Function to increase brightness of an image
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = np.where((255 - v) < value, 255, v + value)
    final_hsv = cv2.merge((h, s, v))
    img_brightened = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img_brightened

# Function to find face encodings in images
def find_encoding(images_list):
    encode_list = []
    for _, img_rgb in images_list:
        # Convert RGB to BGR (as required by face_recognition)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Detect faces in the image
        face_locations = face_recognition.face_locations(img_rgb)

        if len(face_locations) > 0:
            # If faces are detected, encode them
            encode = face_recognition.face_encodings(img_rgb, face_locations)[0]
            encode_list.append(encode)
        else:
            print(f"No face detected in one of the images.")
    return encode_list


@app.route('/')
def home():
    names, rolls, times,majors, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls,majors=majors, times=times, l=l, totalreg=totalreg(),datetoday2=datetoday2)


@app.route('/start')
def start_attendance():
    # Capture and process attendance using face recognition
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    imgBackground = cv2.imread('Resources/background.png')

    folderModePath = 'Resources/Modes'
    modePathList = os.listdir(folderModePath)
    imgModeList = []
    for path in modePathList:
        imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
    print(len(imgModeList))

    # Load face encodings from file
    with open('EncodeFile.p', 'rb') as file:
        encodeListKnownWithIds = pickle.load(file)
    encodeListKnown, studentIds = encodeListKnownWithIds
    print(studentIds)

    marked_student = None

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurrFrame = face_recognition.face_locations(imgS)
        encodeCurrFrame = face_recognition.face_encodings(imgS, faceCurrFrame)

        imgBackground[162:162 + 480, 55:55 + 640] = img
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[0]

        for encode, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
            if encode is not None:
                matches = face_recognition.compare_faces(encodeListKnown, encode)
                faceDist = face_recognition.face_distance(encodeListKnown, encode)
                print("matches:", matches)
                print("FaceDist:", faceDist)

                matchIndex = np.argmin(faceDist)
                print("Match Index:", matchIndex)
                print("Student ID:", studentIds[matchIndex])

                if matches[matchIndex]:
                    # Update Firebase with current time for the matched student
                    ref = db.reference("Students")
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ref.child(studentIds[matchIndex]).update({'time': current_time})

                    # Update attendance record
                    update_attendance(studentIds[matchIndex])

                    # Fetch information of the matched student
                    matched_name = ref.child(studentIds[matchIndex]).child('name').get()
                    matched_roll = studentIds[matchIndex]
                    matched_time = current_time
                    matched_major = ref.child(studentIds[matchIndex]).child('major').get()

                    marked_student = {
                        'name': matched_name,
                        'roll': matched_roll,
                        'time': matched_time,
                        'major': matched_major
                    }

                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)  # Adjust rt for corner radius

        cv2.imshow("Face Attendance", imgBackground)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if marked_student:
        names, rolls, times, majors, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, majors=majors, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2)
    else:
        return redirect('/')  # Redirect to home page after capturing attendance

@app.route('/camera/<newuserid>', methods=['GET'])
def camera(newuserid):
    camera = cv2.VideoCapture(0)  # Initialize camera here
    return render_template('camera.html', userid=newuserid, camera=camera)

# Add user to Firebase Database
@app.route('/add', methods=['POST', 'GET'])
def adduser():
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        newusermajor = request.form['newusermajor']

        # Save user info in session
        session['newusername'] = newusername
        session['newuserid'] = newuserid
        session['newusermajor'] = newusermajor

        return redirect(url_for('camera', newuserid=newuserid))  # Redirect to camera page
    return render_template('home.html')

# Save Image to Firebase Storage and Perform Encoding of the image
@app.route('/save_photo', methods=['POST'])
def save_photo():
    try:
        photo_data = request.form['photo']
        newuserid = request.form['userid']

        # Decode the base64 image data
        data_url_pattern = re.compile(r'data:image/(png|jpeg);base64,(.*)$')
        img_data = data_url_pattern.match(photo_data).group(2)
        img_data = base64.b64decode(img_data)

        # Upload the image to Firebase Storage
        blob = bucket.blob(f'passport_photos/{newuserid}.png')
        blob.upload_from_string(img_data, content_type='image/png')

        # Save user data in Firebase Realtime Database
        ref = db.reference("Students")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            'name': session['newusername'],
            'major': session['newusermajor'],
            'time': current_time
        }
        ref.child(newuserid).set(data)

        # Fetch images from Firebase Storage
        folder_path = 'passport_photos'
        blob_list = bucket.list_blobs(prefix=folder_path)

        # List to store images
        img_list = []
        image_names = []
        student_ids = []

        for blob in blob_list:
            if blob.name.endswith('.png'):  # Ensure only PNG images are processed
                image_name = os.path.basename(blob.name)
                image_names.append(image_name)
                print(f"Fetching image: {image_name}")

                # Extract the ID from the filename
                file_id = os.path.splitext(image_name)[0]
                student_ids.append(file_id)

                # Download blob as bytes
                img_bytes = blob.download_as_bytes()

                # Convert bytes to numpy array
                np_arr = np.frombuffer(img_bytes, np.uint8)

                # Decode numpy array to image
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img is not None:
                    # Increase brightness of the image
                    img_brightened = increase_brightness(img)

                    # Convert BGR to RGB
                    img_rgb = cv2.cvtColor(img_brightened, cv2.COLOR_BGR2RGB)
                    img_list.append((image_name, img_rgb))
                else:
                    print(f"Unable to decode image: {image_name}")

        # Find encodings for detected faces in each image
        print("Encoding Started")
        encode_list_known = find_encoding(img_list)
        encode_list_known_with_id = [encode_list_known, student_ids[:len(encode_list_known)]]  # Ensure matching lengths
        print(encode_list_known_with_id)
        print("Encoding Finished")

        # Save face encodings to file using pickle
        with open('EncodeFile.p', 'wb') as file:
            pickle.dump(encode_list_known_with_id, file)

        # Clear session data
        session.pop('newusername', None)
        session.pop('newuserid', None)
        session.pop('newusermajor', None)

        return redirect('/')

    except Exception as e:
        print(f"Error occurred: {e}")
        return "Error occurred while saving photo"


if __name__ == "__main__":
    app.run(debug=True)