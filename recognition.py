import os
import json
import torch
import psycopg2
import numpy as np
import cv2
import time
from queue import Queue
from threading import Thread
from PIL import Image
from scipy.spatial.distance import cosine
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Load models
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device, min_face_size=40, thresholds=[0.8, 0.9, 0.9])

# Connect to database
conn = psycopg2.connect(dbname="attendance_db", user="postgres", password="postgress123", host="localhost")
cursor = conn.cursor()

cursor.execute("SELECT student_id, name, department, level, embedding FROM students")
stored_data = cursor.fetchall()

# Cache detected faces
detected_faces_cache = {}

def recognize_face(new_embedding, stored_embedding):
    """Calculate similarity score between embeddings."""
    new_embedding = np.array(new_embedding).flatten()
    stored_embedding = np.array(stored_embedding).flatten()
    return 1 - cosine(new_embedding, stored_embedding)

def check_previous_faces(new_embedding):
    """Check if face matches a cached one within recent frames."""
    current_time = time.time()
    
    for student_id, data in detected_faces_cache.items():
        stored_embedding = data["embedding"]
        score = recognize_face(new_embedding, stored_embedding)

        if score > 0.75 and (current_time - data["timestamp"]) < 5:  # 5 seconds validity
            return student_id  
    return None  

def update_face_cache(student_id, new_embedding):
    """Update cache with a newly detected face."""
    detected_faces_cache[student_id] = {
        "embedding": new_embedding,
        "timestamp": time.time()
    }

def detect_motion(prev_frame, frame):
    """Detect movement by comparing consecutive frames."""
    diff = cv2.absdiff(prev_frame, frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    return np.sum(thresh) > 5000  # Adjust threshold for sensitivity

def process_frame(frame, prev_frame):
    """Process a single frame for face recognition."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    if prev_frame is not None and not detect_motion(prev_frame, frame):
        return [], frame_rgb  # Skip processing if no movement detected

    face_tensors = mtcnn(img)
    recognized_students = []

    if face_tensors is not None:
        for face_tensor in face_tensors:
            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                new_embedding = resnet(face_tensor).cpu().squeeze().tolist()

            matched_id = check_previous_faces(new_embedding)

            if matched_id:
                print(f"Using cached result: {matched_id}")
                for student_id, name, department, level, stored_embedding_json in stored_data:
                    if student_id == matched_id:
                        recognized_students.append({
                            "ID": student_id,
                            "Name": name,
                            "Department": department,
                            "Level": level,
                            "Score": 0.80  # Cached confidence
                        })
                        break
            else:
                for student_id, name, department, level, stored_embedding_json in stored_data:
                    stored_embedding = np.array(stored_embedding_json).flatten().tolist()
                    score = recognize_face(new_embedding, stored_embedding)

                    if score > 0.65:
                        update_face_cache(student_id, new_embedding)
                        recognized_students.append({
                            "ID": student_id,
                            "Name": name,
                            "Department": department,
                            "Level": level,
                            "Score": round(score, 3)
                        })
                        break

    return recognized_students, frame_rgb

def draw_info(frame, students):
    """Draw recognition information on the frame."""
    for i, student in enumerate(students):
        text = f"{student['Name']} ({student['ID']}) - {student['Department']}"
        cv2.putText(frame, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def capture_frames(cap, frame_queue):
    """Thread function to continuously read frames from the camera."""
    while True:
        ret, frame = cap.read()
        if ret:
            frame_queue.put(frame)

def main():
    stream_url = "https://10.89.71.159:8080/video"
    cap = cv2.VideoCapture(stream_url)  
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for better speed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce lag due to buffering

    print("Starting live stream recognition... Press 'q' to quit.")
    
    frame_queue = Queue()
    capture_thread = Thread(target=capture_frames, args=(cap, frame_queue))
    capture_thread.daemon = True
    capture_thread.start()
    
    prev_frame = None

    while True:
        if frame_queue.qsize() > 0:
            frame = frame_queue.get()
            recognized_students, frame_rgb = process_frame(frame, prev_frame)
            prev_frame = frame  # Store previous frame for motion detection

            frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if recognized_students:
                frame_display = draw_info(frame_display, recognized_students)
                print("\nRecognized:")
                for student in recognized_students:
                    print(f"{student['Name']} (ID: {student['ID']}) - Score: {student['Score']}")

            cv2.imshow('Face Recognition', frame_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()
