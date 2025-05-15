import os
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt

# Initialize MTCNN (face detector)
mtcnn = MTCNN(
    keep_all=True,          # Detect all faces in the image
    min_face_size=40,       # Minimum face size to detect (smaller = slower but more thorough)
    thresholds=[0.6, 0.7, 0.7],  # Detection thresholds (lower = more sensitive)
    device='cpu'            # Use 'cuda' if you have a GPU
)

# Paths
image_folder = "./FE21A204/"  
output_folder = "detected_faces"
os.makedirs(output_folder, exist_ok=True)

# Process each image
for img_name in os.listdir(image_folder):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # Skip non-image files

    img_path = os.path.join(image_folder, img_name)
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_name}")
        continue
    
    # Convert to RGB (MTCNN expects PIL images in RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Detect faces and landmarks
    boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)
    
    # Draw results if faces found
    if boxes is not None:
        for box, landmark in zip(boxes, landmarks):
            # Draw bounding box (green)
            cv2.rectangle(
                img, 
                (int(box[0]), int(box[1])),  # Top-left corner
                (int(box[2]), int(box[3])),  # Bottom-right corner
                (0, 255, 0),  # Green color
                2  # Line thickness
            )
            
            # Draw facial landmarks (red dots)
            for point in landmark:
                cv2.circle(
                    img, 
                    (int(point[0]), int(point[1])), 
                    2,  # Radius
                    (0, 0, 255),  # Red color
                    -1  # Filled circle
                )
    
    # Save output
    output_path = os.path.join(output_folder, f"detected_{img_name}")
    cv2.imwrite(output_path, img)
    print(f"Processed: {img_name}")

print(f"\nDone! Detected faces saved to '{output_folder}'.")