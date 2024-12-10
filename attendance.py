import cv2
import os
import pandas as pd
import streamlit as st
from datetime import datetime

# File paths
FACE_DATA_PATH = "registered_faces"
ATTENDANCE_FILE = "attendance.csv"
SESSION_ATTENDANCE_FILE = "session_attendance.txt"

# Initialize the Haar Cascade (Viola-Jones Algorithm)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def register_face(name, roll_number):
    """
    Register a new face by capturing an image from the webcam.
    """
    if not os.path.exists(FACE_DATA_PATH):
        os.makedirs(FACE_DATA_PATH)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        st.error("Error: Unable to access the webcam.")
        return

    st.info("Look into the camera to register your face. Press 's' to save and 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Error: Failed to read from the webcam.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y + h, x:x + w]
                filename = f"{name}_{roll_number}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                cv2.imwrite(os.path.join(FACE_DATA_PATH, filename), face_roi)
                st.success(f"Face registered for {name} (Roll No: {roll_number}).")
            break
        elif key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


def monitor_classroom():
    """
    Monitor the classroom and mark attendance for detected faces.
    """
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        st.error("Error: Unable to access the webcam.")
        return

    st.info("Monitoring classroom. Press 'q' to stop.")

    face_data = {}
    for file in os.listdir(FACE_DATA_PATH):
        if file.endswith(".jpg"):
            name, roll_number, _ = file.split("_")
            img_path = os.path.join(FACE_DATA_PATH, file)
            face_data[(name, roll_number)] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    session_attendance = set()  # To track faces during this session

    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Error: Failed to read from the webcam.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            roi = gray_frame[y:y + h, x:x + w]
            recognized_name = "Unknown"
            recognized_roll_number = ""

            for (name, roll_number), stored_face in face_data.items():
                resized_stored_face = cv2.resize(stored_face, (w, h))
                diff = cv2.absdiff(roi, resized_stored_face)
                score = cv2.mean(diff)[0]
                if score < 50:  # Adjust threshold as needed
                    recognized_name = name
                    recognized_roll_number = roll_number
                    session_attendance.add((name, roll_number))
                    break

            cv2.putText(frame, f"{recognized_name} ({recognized_roll_number})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Monitor Classroom", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    # Subject Selection
    st.write("Select the subject to mark attendance:")
    subjects = ["CS", "Java", "Python", "Mathematics", "DBMS"]
    subject = st.selectbox("Subjects", subjects)

    if st.button("Mark Attendance for Selected Subject"):
        for name, roll_number in session_attendance:
            mark_attendance(name, roll_number, subject)
        st.success(f"Attendance marked for subject: {subject}")


def mark_attendance(name, roll_number, subject):
    """
    Marks the attendance of a recognized face by appending to the attendance CSV file.
    """
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Create a new DataFrame for the attendance
    df = pd.DataFrame([[name, roll_number, subject, current_time]], columns=["Name", "Roll Number", "Subject", "Time"])

    # Append to the existing attendance file (or create a new one if it doesn't exist)
    if os.path.exists(ATTENDANCE_FILE):
        df.to_csv(ATTENDANCE_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(ATTENDANCE_FILE, mode='w', header=True, index=False)


def view_attendance():
    """
    Display attendance for the current session along with images of present students.
    """
    st.write("Attendance Records:")
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)
    except FileNotFoundError:
        st.warning("Attendance file not found. Please monitor the classroom first.")


# Streamlit interface
def main():
    st.title("Face Recognition Attendance System")

    menu = ["Register Face", "Monitor Classroom", "View Attendance"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Register Face":
        name = st.text_input("Enter Name", max_chars=50)
        roll_number = st.text_input("Enter Roll Number", max_chars=10)
        if st.button("Register"):
            if name.strip() and roll_number.strip():
                register_face(name.strip(), roll_number.strip())
            else:
                st.error("Name and Roll Number cannot be empty.")
    elif choice == "Monitor Classroom":
        if st.button("Start Monitoring"):
            monitor_classroom()
    elif choice == "View Attendance":
        if st.button("View Attendance"):
            view_attendance()


if __name__ == "__main__":
    main()
