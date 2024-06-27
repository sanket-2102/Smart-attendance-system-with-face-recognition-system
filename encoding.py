import cv2
import face_recognition
import pickle
import os

folderPath = 'local_images'
modePathList = os.listdir(folderPath)
imgList = []
studentIds = []

# Load images and their corresponding student IDs
for path in modePathList:
    img = cv2.imread(os.path.join(folderPath, path))
    if img is not None:
        imgList.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        studentIds.append(os.path.splitext(path)[0])
    else:
        print(f"Unable to read image: {path}")

# Function to find face encodings in a list of images
def findEncodings(imgList):
    encodingList = []
    for img in imgList:
        encode = face_recognition.face_encodings(img)
        if len(encode) > 0:
            encodingList.append(encode[0])
        else:
            print(f"No face detected in image.")
            encodingList.append(None)  # Handle cases where no face is detected
    return encodingList

# Generate encodings for each image
print("Encoding Started")
encodeListKnown = findEncodings(imgList)
encodingListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

# Save encodings to a pickle file
file_path = "EncodeFile.p"
with open(file_path, "wb") as file:
    pickle.dump(encodingListKnownWithIds, file)

print(f"Encodings saved to {file_path}")
