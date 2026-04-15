import cv2
import os
import argparse
import time
import shutil

def process_and_add_images(input_dir, class_name, is_background=False):
    output_dir = f"dataset/{class_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Directory exists: {output_dir}. Appending new images.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(files)} images in '{input_dir}'. Processing for label '{class_name}'...")
    
    success_count = 0
    
    for filename in files:
        img_path = os.path.join(input_dir, filename)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Warning: Could not read {filename}. Skipping.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = []
        if is_background:
            h, w = frame.shape[:2]
            # Use multiple crops or just center crop? 
            # Collector uses one center crop for background.
            crop_size = min(h, w, 300) # Ensure crop isn't larger than image
            start_y = max(0, (h - crop_size) // 2)
            start_x = max(0, (w - crop_size) // 2)
            faces = [(start_x, start_y, crop_size, crop_size)]
        else:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print(f"No face detected in {filename}. Skipping.")
            continue
            
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            if face_img.size > 0:
                face_resized = cv2.resize(face_img, (96, 96))

                timestamp = int(time.time() * 1000)
                # Add a small random suffix or counter to avoid overwriting if processing is too fast
                # But typically loop is slow enough, or we can just use a counter.
                # Let's use timestamp + loop index to be safe if iterating fast
                out_filename = f"{timestamp}_{success_count}.jpg"
                file_path = os.path.join(output_dir, out_filename)
                
                cv2.imwrite(file_path, face_resized)
                success_count += 1
    
    print(f"Processing complete. Added {success_count} images to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process folder of images for ESP32 face dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input folder containing images")
    parser.add_argument("--label", type=str, required=True, help="Label/Class name for the images")
    parser.add_argument("--background", action="store_true", help="Process as background (no face detection)")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
    else:
        process_and_add_images(args.input_dir, args.label, args.background)
