import os
import json
import time
import torch
import psycopg2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import datasets
from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Set CPU threads if not using GPU
if device.type == 'cpu':
    torch.set_num_threads(4)  

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

conn = psycopg2.connect(dbname="attendance_db", user="postgres", password="postgress123", host="localhost")
cursor = conn.cursor()

student_data = pd.read_csv("studentsName.csv")
student_info = {row["student_id"]: {"name": row["name"], "department": row["department"], "level": row["level"]}
                for _, row in student_data.iterrows()}

detected_faces_path = "detectedFaces/"
dataset_path = "dataset/"

def process_folder_batch(folder_path, student_id, batch_size=4):
    """Process all images for one person in batches"""
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_tensors = []
        
        for img_file in batch_files:
            img_path = os.path.join(folder_path, img_file)
            try:
                img = Image.open(img_path)
                img_array = np.array(img).astype(np.float32) / 255.0
                face_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0) #this is rearranging to match what the model expects
                batch_tensors.append(face_tensor)
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue
        
        if batch_tensors:
            try:
                batch = torch.cat(batch_tensors).to(device)
                with torch.no_grad():
                    embeddings = resnet(batch).cpu()
                    all_embeddings.append(embeddings)
                del batch
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue
    
    if all_embeddings:
        avg_embedding = torch.mean(torch.cat(all_embeddings), dim=0)
        return avg_embedding.tolist()
    return None

def detect_and_save_faces():
    """Run face detection if needed"""
    print("No detected faces found. Running full pipeline...")
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=True,
        device=device,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7]
    )
    
    os.makedirs(detected_faces_path, exist_ok=True)
    workers = 0 if os.name == 'nt' else 4

    class NamedImageFolder(datasets.ImageFolder):
        def __getitem__(self, index):
            original_tuple = super().__getitem__(index)
            path = self.samples[index][0]
            return original_tuple + (path,)

    dataset = NamedImageFolder(dataset_path)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=lambda x: x[0], num_workers=workers)

    file_counter = {}

    for img, label, img_path in tqdm(loader, desc="Detecting faces"):
        faces, probs = mtcnn(img, return_prob=True)
        if faces is not None:
            student_id = dataset.idx_to_class[label]
            
            if student_id not in file_counter:
                file_counter[student_id] = 0
                os.makedirs(os.path.join(detected_faces_path, student_id), exist_ok=True)
            
            for i, (face, prob) in enumerate(zip(faces, probs)):
                if prob < 0.7:
                    continue
                    
                student_faces_folder = os.path.join(detected_faces_path, student_id)
                original_name = os.path.splitext(os.path.basename(img_path))[0]
                save_name = f"{original_name}_{file_counter[student_id]}.jpg"
                file_counter[student_id] += 1

                img_array = face.permute(1, 2, 0).detach().cpu().numpy()
                img_array = (img_array * 255).astype(np.uint8)
                Image.fromarray(img_array).save(os.path.join(student_faces_folder, save_name))

    print(f"✅ Face detection completed. Saved {sum(file_counter.values())} faces.")

# Check if we need face detection
if not os.path.exists(detected_faces_path) or not any(os.listdir(detected_faces_path)):
    detect_and_save_faces()
print("\nProcessing student folders...")
for student_id in tqdm(os.listdir(detected_faces_path), desc="Students"):
    student_folder = os.path.join(detected_faces_path, student_id)
    if not os.path.isdir(student_folder):
        continue
    
    embedding = process_folder_batch(student_folder, student_id, batch_size=2)
    
    if embedding is None:
        print(f" No valid faces found for {student_id}")
        continue
    
    student_details = student_info.get(student_id, {
        "name": "Unknown",
        "department": "Unknown",
        "level": "Unknown"
    })
    try:
        cursor.execute("""
            INSERT INTO students (student_id, name, department, level, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (student_id) DO UPDATE SET
                name = EXCLUDED.name,
                department = EXCLUDED.department,
                level = EXCLUDED.level,
                embedding = EXCLUDED.embedding
        """, (student_id, 
              student_details["name"],
              student_details["department"],
              student_details["level"],
              json.dumps(embedding)))
        
        conn.commit()
    except Exception as e:
        print(f" Database error for {student_id}: {str(e)}")
        conn.rollback()

print(" All student embeddings processed and stored!")
conn.close()