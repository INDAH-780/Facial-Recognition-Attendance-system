import os
import mmcv
import cv2
import torch
import numpy as np
import psycopg2
from PIL import Image, ImageDraw
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Load Models
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40, thresholds=[0.8, 0.9, 0.9])

# Connect to Database
conn = psycopg2.connect(dbname="attendance_db", user="postgres", password="postgress123", host="localhost")
cursor = conn.cursor()

# Load stored embeddings
cursor.execute("SELECT student_id, name, department, level, embedding FROM students")
stored_data = cursor.fetchall()

# Recognition Function
def recognize_face(new_embedding, stored_embedding):
    """Calculate similarity score between embeddings."""
    new_embedding = np.array(new_embedding).flatten()
    stored_embedding = np.array(stored_embedding).flatten()
    return 1 - np.dot(new_embedding, stored_embedding) / (
        np.linalg.norm(new_embedding) * np.linalg.norm(stored_embedding)
    )

# Function to log attendance
def log_attendance(student_id, name):
    """Log attendance in the database."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        cursor.execute(
            "INSERT INTO attendance (student_id, name, timestamp) VALUES (%s, %s, %s) "
            "ON CONFLICT (student_id) DO UPDATE SET timestamp = EXCLUDED.timestamp",
            (student_id, name, timestamp)
        )
        conn.commit()
        print(f"✅ Attendance logged for {name} at {timestamp}")
    except psycopg2.Error as e:
        conn.rollback()
        print(f"❌ Attendance logging error: {str(e)}")

# Function to process video with tracking
def process_video(video_path):
    """Reads video file, detects faces, recognizes students, and saves results."""
    video = mmcv.VideoReader(video_path)
    frames_tracked = []

    for i, frame in enumerate(video):
        print(f'\rTracking frame: {i + 1}', end='')

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(pil_image)

        if boxes is None:
            continue  # Skip frames without detected faces

        frame_draw = pil_image.copy()
        draw = ImageDraw.Draw(frame_draw)
        recognized_students = []

        for box in boxes:
            left, top, right, bottom = map(int, box)

            # ✅ Ensure detected face is large enough
            if right - left < 10 or bottom - top < 10:
                continue  # Skip tiny faces

            # ✅ Resize face to model’s expected size (160x160)
            face = pil_image.crop((left, top, right, bottom)).resize((160, 160), Image.BILINEAR)
            face_tensor = torch.tensor(np.array(face)).permute(2, 0, 1).unsqueeze(0).float().to(device)

            with torch.no_grad():
                new_embedding = resnet(face_tensor).cpu().squeeze().tolist()

            # ✅ Store similarity scores for proper comparison
            possible_matches = []
            for student_id, name, department, level, stored_embedding_json in stored_data:
                stored_embedding = np.array(stored_embedding_json).flatten().tolist()
                score = recognize_face(new_embedding, stored_embedding)

                print(f"🔎 Similarity Score with {name}: {score}")  # Debugging step

                if score > 0.85:  # Stricter threshold
                    possible_matches.append((name, score, student_id, department, level))

            # ✅ Sort matches by highest score and select the best match
            if possible_matches:
                best_match = sorted(possible_matches, key=lambda x: x[1], reverse=True)[0]
                recognized_students.append({
                    "ID": best_match[2],
                    "Name": best_match[0],
                    "Department": best_match[3],
                    "Level": best_match[4],
                    "Score": round(best_match[1], 3)
                })
                log_attendance(best_match[2], best_match[0])
                draw.rectangle(box.tolist(), outline="red", width=6)
                draw.text((left, top - 15), best_match[0], fill="red")

        # ✅ Only print logs when a face is detected and recognized
        if recognized_students:
            print(f"✅ Frame Processed - Recognized Students: {recognized_students}")

        frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))

    print("\nVideo Processing Done")

    # Save tracked video with detected faces
    dim = frames_tracked[0].size
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    video_tracked = cv2.VideoWriter('video_tracked.mp4', fourcc, 25.0, dim)

    for frame in frames_tracked:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    
    video_tracked.release()
    print("✅ Saved Tracked Video with Bounding Boxes")

# Select Input Type
video_path = "test.mp4"  # Replace with actual video file path
process_video(video_path)

conn.close()
