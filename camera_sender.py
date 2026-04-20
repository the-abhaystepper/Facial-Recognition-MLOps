import argparse
import time
import threading
import csv
import os
from datetime import datetime
from queue import Queue
import cv2
import requests
import subprocess
from cassandra.cluster import Cluster

# Initialize Cassandra connection for real-time distributed logging
try:
    cluster = Cluster(['127.0.0.1'])
    cassandra_session = cluster.connect('security_system')
    print("Connected to Distributed Database (Cassandra)")
except Exception as e:
    cassandra_session = None
    print(f"Warning: Cassandra not reached. Dashboard updates via manual ingest only.")

# Global queue and state
send_queue = Queue(maxsize=1)
last_prediction = "Waiting..."

def sender_worker(url):
    """Background thread to send frames and read the prediction result."""
    global last_prediction
    while True:
        data = send_queue.get()
        if data is None: break
        
        try:
            # Re-enabling the 10s timeout for network stability
            response = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'}, timeout=10)
            if response.status_code == 200:
                response_text = response.text.strip()
                last_prediction = response_text
                
                # Log to CSV
                try:
                    parts = response_text.split(':')
                    label = parts[0]
                    confidence = parts[1] if len(parts) > 1 else "0.0"
                    
                    with open("detection_log.csv", "a", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), label, confidence])
                    
                    # Log to Distributed DB (Cassandra) in Real-time
                    if cassandra_session:
                        try:
                            ts = datetime.now()
                            # 1. Log to main history
                            query1 = "INSERT INTO detections (timestamp, label, confidence) VALUES (?, ?, ?)"
                            prepared1 = cassandra_session.prepare(query1)
                            cassandra_session.execute(prepared1, (ts, label, float(confidence)))
                            
                            # 2. Log to Live Feed
                            query2 = "INSERT INTO recent_activity (feed_type, timestamp, label, details) VALUES (?, ?, ?, ?)"
                            prepared2 = cassandra_session.prepare(query2)
                            cassandra_session.execute(prepared2, ("LIVE_FEED", ts, label, f"Confidence: {confidence}"))
                        except Exception as e:
                            print(f"Cassandra Live Log Error: {e}")
                except Exception as e:
                    print(f"Logging error: {e}")
        except Exception as e:
            if "last_err_print" not in globals() or time.time() - last_err_print > 5:
                print(f"Connection Error: {e}")
                globals()["last_err_print"] = time.time()
        finally:
            send_queue.task_done()

def anomaly_watcher(cassandra_session):
    """Watches event_log.csv for new anomalies and pushes them to Cassandra."""
    if not cassandra_session: return
    last_pos = os.path.getsize("event_log.csv") if os.path.exists("event_log.csv") else 0
    
    while True:
        if os.path.exists("event_log.csv"):
            with open("event_log.csv", "r") as f:
                f.seek(last_pos)
                new_data = f.read()
                last_pos = f.tell()
                for line in new_data.splitlines():
                    try:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            ts, ev_type, msg = parts[0], parts[1], parts[2] if len(parts)>2 else ""
                            q = "INSERT INTO recent_activity (feed_type, timestamp, label, details) VALUES (?, ?, ?, ?)"
                            cassandra_session.execute(cassandra_session.prepare(q), ("LIVE_FEED", datetime.now(), ev_type, msg))
                    except: pass
        time.sleep(2)

def main():
    parser = argparse.ArgumentParser(description="Laptop Camera Sender to ESP32")
    parser.add_argument("--ip", type=str, default="esp32.local", help="IP address or hostname of the ESP32")
    args = parser.parse_args()

    url = f"http://{args.ip}/upload"
    print(f"Streaming to {url}...")
    
    # Initialize CSV header if needed
    if not os.path.exists("detection_log.csv"):
        with open("detection_log.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Label", "Confidence"])

    threading.Thread(target=sender_worker, args=(url,), daemon=True).start()
    threading.Thread(target=anomaly_watcher, args=(cassandra_session,), daemon=True).start()

    # Start R Supervisory System
    print("Starting R Supervisory System...")
    r_process = None
    try:
        r_process = subprocess.Popen([r"C:\Program Files\R\R-4.5.2\bin\x64\Rscript.exe", "supervisory_system.R"])
    except Exception as e:
        print(f"Warning: Could not start R script: {e}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Press 'q' to quit.")

    
    # Track previous state to avoid spamming requests
    state_lock = threading.Lock()
    state = {
        "lockdown": "0",
        "message": "System Active",
        "last_lockdown_sent": "0",
        "last_msg_sent": ""
    }

    def signal_worker():
        while True:
            try:
                # 1. Check Lockdown Signal
                if os.path.exists("lockdown_signal.txt"):
                    with open("lockdown_signal.txt", "r") as f:
                        curr_lockdown = f.read().strip()
                    
                    with state_lock:
                        last_sent = state["last_lockdown_sent"]
                    
                    if curr_lockdown != last_sent:
                        try: 
                            requests.get(f"http://{args.ip}/set_lockdown?state={curr_lockdown}", timeout=1.0)
                            with state_lock: state["last_lockdown_sent"] = curr_lockdown
                            print(f"Lockdown synchronized: {curr_lockdown}")
                        except: pass
                
                # 2. Check Supervisory Message
                if os.path.exists("supervisory_msg.txt"):
                    with open("supervisory_msg.txt", "r") as f:
                        curr_msg = f.read().strip()
                    
                    with state_lock:
                        last_sent_msg = state["last_msg_sent"]
                    
                    if curr_msg != last_sent_msg and curr_msg:
                        try:
                            requests.get(f"http://{args.ip}/set_message", params={'msg': curr_msg[:30]}, timeout=1.0)
                            with state_lock: state["last_msg_sent"] = curr_msg
                        except: pass
            except Exception:
                pass
            time.sleep(0.5) # Poll every half second

    threading.Thread(target=signal_worker, daemon=True).start()

    while True:
        # The main while loop is now clean and only handles camera/UI

        ret, frame = cap.read()
        if not ret: break
        
        small_gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))
        
        frame_to_send = None
        
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = int(fx*2), int(fy*2), int(fw*2), int(fh*2)
            
            x, y = max(0, x), max(0, y)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {last_prediction}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_img = gray_full[y:y+h, x:x+w]
            
            if face_img.size > 0:
                target_img = cv2.resize(face_img, (96, 96))
                frame_to_send = target_img.tobytes()
        else:
            cv2.putText(frame, "Searching for Face...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Camera Feed", frame)

        if frame_to_send and send_queue.empty():
            try:
                send_queue.put_nowait(frame_to_send)
            except: pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if r_process:
        print("Signaling R Supervisory System to stop...")
        with open("supervisory_stop.txt", "w") as f: 
            f.write("STOP")
            f.flush()
            os.fsync(f.fileno()) # Force write to disk for R to see it
        
        print("Waiting for R to generate graphs... (Look for the R plot window)")
        
        # Wait for R, but check if it's still alive
        while r_process.poll() is None:
            time.sleep(0.5)
            # Check if user deleted the stop file manually as an 'emergency exit'
            if not os.path.exists("supervisory_stop.txt"):
                break
            
    print("\nStreaming stopped.")

if __name__ == "__main__":
    main()
