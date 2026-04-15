import cv2
import numpy as np
import requests
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="ESP32 OV7670 Viewer")
    parser.add_argument("--ip", type=str, default="esp32.local", help="IP address or hostname of the ESP32 (default: esp32.local)")
    args = parser.parse_args()

    url = f"http://{args.ip}/capture"
    print(f"Connecting to {url}...")
    print("Press 'q' to quit.")

    while True:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                img_array = np.array(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, -1)
                
                if img is not None:
                    img_resized = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("ESP32 OV7670 Feed", img_resized)
                else:
                    print("Failed to decode image")
            else:
                print(f"Server returned status code {response.status_code}")
        
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
