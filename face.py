import cv2
import face_recognition
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = face_recognition.load_image_file(img_path)
        images.append(image)
    return images

def create_face_encodings(images):
    face_encodings = []
    for image in images:
        encoding = face_recognition.face_encodings(image)[0]
        face_encodings.append(encoding)
    return face_encodings

# Specify the folder containing images of each person
person1_folder = "faces\Amir Salah"
person2_folder = "faces\Mohamed Hussain"
person3_folder = "faces\Omar Ahmed"
#person4_folder = "faces\George"

# Load images for each person
person1_images = load_images_from_folder(person1_folder)
person2_images = load_images_from_folder(person2_folder)
person3_images = load_images_from_folder(person3_folder)
#person4_images = load_images_from_folder(person4_folder)

# Create face encodings for each person
person1_encodings = create_face_encodings(person1_images)
person2_encodings = create_face_encodings(person2_images)
person3_encodings = create_face_encodings(person3_images)
#person4_encodings = create_face_encodings(person4_images)

# Store the encodings and corresponding labels for training
encodings = person1_encodings + person2_encodings + person3_encodings + person4_encodings
labels = ["Amir Salah"] * len(person1_encodings) + ["Mohamed Hussain"] * len(person2_encodings) + ["Omar Ahmed"] * len(person3_encodings) #+ ["George"] * len(person4_encodings)

# Load the trained model or train the model here (using a suitable machine learning library)

# Open the webcam
video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera, you can change it if you have multiple cameras

while True:
    # Capture each frame from the webcam
    ret, frame = video_capture.read()

    # Find face locations in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Perform face recognition for each face in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Replace this part with your face recognition model prediction
        # For simplicity, we are using the first label for the example
        label = "Unknown"
        if len(encodings) > 0:
            results = face_recognition.compare_faces(encodings, face_encoding)
            if True in results:
                index = results.index(True)
                label = labels[index]

        # Draw a rectangle around the face and display the label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the frame with faces and recognition results
    cv2.imshow("Face Recognition", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()
