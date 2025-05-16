import os
import json
import torch
import psycopg2
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.distance import cosine
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(
    image_size=160, margin=20, keep_all=True, device=device,
    min_face_size=40, thresholds=[0.8, 0.9, 0.9]
)

conn = psycopg2.connect(dbname="attendance_db", user="postgres", password="postgress123", host="localhost")
cursor = conn.cursor()

cursor.execute("SELECT student_id, name, department, level, embedding FROM students")
stored_data = cursor.fetchall()

def recognize_face(new_embedding, stored_embedding):
    """Calculate similarity score between embeddings."""
    new_embedding = np.array(new_embedding).flatten()
    stored_embedding = np.array(stored_embedding).flatten()
    return 1 - cosine(new_embedding, stored_embedding)

def process_frame(frame):
    """Process a single frame for face recognition."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    

    face_tensors = mtcnn(img)
    
    recognized_students = []
    
    if face_tensors is not None:
        for face_tensor in face_tensors:
            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                new_embedding = resnet(face_tensor).cpu().squeeze().tolist()
            
            for student_id, name, department, level, stored_embedding_json in stored_data:
                stored_embedding = np.array(stored_embedding_json).flatten().tolist()
                score = recognize_face(new_embedding, stored_embedding)
                
                if score > 0.65:
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
        cv2.putText(frame, text, (10, 30 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def main():
    stream_url = "https://192.168.88.31:8080/video"
    
    cap = cv2.VideoCapture(stream_url)  
    
  
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Starting live stream recognition... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        recognized_students, frame_rgb = process_frame(frame)
        
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