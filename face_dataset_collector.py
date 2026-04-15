import cv2
import os
import argparse
import time

def create_dataset(class_name, num_samples=100, is_background=False):
    output_dir = f"dataset/{class_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Directory exists: {output_dir}. Appending new images.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    count = 0
    print(f"Collecting {num_samples} samples for '{class_name}'...")
    if is_background:
        print("Note: Running in BACKGROUND mode (No face detection). Capturing center crops.")
    print("Press 's' to start/pause saving, 'q' to quit.")

    saving = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if is_background:
            h, w = frame.shape[:2]
            crop_size = 300 
            start_y = max(0, (h - crop_size) // 2)
            start_x = max(0, (w - crop_size) // 2)
            faces = [(start_x, start_y, crop_size, crop_size)]
        else:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            color = (0, 255, 0) if saving else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            if saving and count < num_samples:
                face_img = gray[y:y+h, x:x+w]
                if face_img.size > 0:
                    face_resized = cv2.resize(face_img, (96, 96))

                    timestamp = int(time.time() * 1000)
                    file_path = f"{output_dir}/{timestamp}.jpg"
                    cv2.imwrite(file_path, face_resized)
                    
                    count += 1
                    time.sleep(0.1) 

        cv2.putText(frame, f"Samples: {count}/{num_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        status_text = "SAVING" if saving else "PAUSED (Press 's')"
        cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if saving else (0, 0, 255), 2)

        cv2.imshow("Dataset Collector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            saving = not saving
        
        if count >= num_samples:
            print("Collection complete!")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect face dataset for ESP32 model")
    parser.add_argument("--name", type=str, required=True, help="Name of the person/class")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to collect")
    parser.add_argument("--background", action="store_true", help="Capture background (no face detection needed)")
    args = parser.parse_args()

    create_dataset(args.name, args.samples, args.background)
