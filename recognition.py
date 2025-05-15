import os
import json
import torch
import psycopg2
import numpy as np
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

# Step 1: Recognition Function
def recognize_face(new_embedding, stored_embedding):
    """Calculate similarity score between embeddings."""
    new_embedding = np.array(new_embedding).flatten()
    stored_embedding = np.array(stored_embedding).flatten()
    return 1 - cosine(new_embedding, stored_embedding)

# Step 2: Load All Stored Embeddings from Database
cursor.execute("SELECT student_id, name, department, level, embedding FROM students")
stored_data = cursor.fetchall()

# Step 3: Load Image for Recognition
image_path = "group.png"  
if not os.path.exists(image_path):
    print(" Image not found. Please provide a valid image path.")
    conn.close()
    exit()

img = Image.open(image_path).convert("RGB")  

face_tensors = mtcnn(img)  

if face_tensors is None:
    print(" No faces detected in the image.")
    conn.close()
    exit()

# Step 4: Generate Embeddings for Each Face
recognized_students = []  

for face_tensor in face_tensors:
    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        new_embedding = resnet(face_tensor).cpu().squeeze().tolist()

    # Step 5: Compare Against Stored Embeddings
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

# Step 6: Display Recognized Students
if recognized_students:
    print(" Recognized Students:")
    for student in recognized_students:
        print(student)
else:
    print(" No faces recognized.")
conn.close()
