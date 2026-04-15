import csv
import time
import random
from datetime import datetime

log_file = "detection_log.csv"

def write_log(label, confidence):
    with open(log_file, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), label, confidence])
    print(f"Logged: {label} ({confidence}%)")

print("Starting Simulation in 2 seconds...")
time.sleep(2)

# 1. Normal Random Data
print("\n--- Phase 1: Normal Random Data ---")
for _ in range(5):
    write_log("Unknown", random.uniform(20, 50))
    time.sleep(0.5)

# 2. Unlock Sequence (Abhay > 85% for 3s)
print("\n--- Phase 2: Unlock Sequence (Abhay) ---")
for _ in range(8): # 8 * 0.5s = 4s > 3s
    write_log("Abhay", random.uniform(86, 99))
    time.sleep(0.5)

# 3. Lockdown Sequence (3 Fails)
print("\n--- Phase 3: Lockdown Sequence (3 Fails) ---")
for i in range(3):
    write_log("Unknown", random.uniform(10, 40))
    time.sleep(0.5)

# 4. Anomaly - High Frequency
print("\n--- Phase 4: High Frequency Anomaly ---")
for _ in range(40):
    write_log("Intruder", random.uniform(10, 30))
    # No sleep or very short to simulate burst
