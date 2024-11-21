import face_recognition
import numpy as np
import cv2
import csv
import os

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Opening CSV file
with open('data - data.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

user_face_encodings = []
known_face_names = []

# Load images and encode faces
for row in data:
    name = row[0]
    image_path = os.path.join("Data", row[1])  # Creates a path using the Data folder
    print(f"Loading image from: {image_path}")  # Print the path for debugging

    try:
        # Use OpenCV to read the image
        image = cv2.imread(image_path)

        # Convert to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        face_locations = face_recognition.face_locations(rgb_image)

        # Only proceed if a face is found
        if face_locations:
            # Get the first face encoding
            encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
            user_face_encodings.append(encoding)
            known_face_names.append(name)
        else:
            print(f"No face found in {image_path}")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture video")
        break

    # Convert the image from BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)

    # Only proceed if faces are found
    if face_locations:
        # Compute face encodings for the detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare faces
            matches = face_recognition.compare_faces(user_face_encodings, face_encoding)
            name = "Random Person"

            # Find the best match
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw rectangle and name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            if name != "Random Person":
                print(f"{name} was detected")

    cv2.imshow('Video', frame)

    # Q to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()