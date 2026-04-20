from flask import Flask, render_template, jsonify
from cassandra.cluster import Cluster
import requests
import os
import datetime

app = Flask(__name__)

# Cassandra connection
def get_cassandra_session():
    try:
        cluster = Cluster(['127.0.0.1'])
        session = cluster.connect('security_system')
        return session
    except:
        return None

# REST API call to MLflow
def get_mlflow_data():
    try:
        response = requests.get("http://localhost:5000/api/2.0/mlflow/experiments/get-by-name?experiment_name=Face_Recognition_Auth", timeout=2)
        if response.status_code == 200:
            exp_id = response.json()['experiment']['experiment_id']
            # SEARCH requires POST in MLflow REST API
            search_payload = {"experiment_ids": [exp_id], "max_results": 1, "order_by": ["attributes.start_time DESC"]}
            runs_resp = requests.post("http://localhost:5000/api/2.0/mlflow/runs/search", json=search_payload, timeout=2)
            
            if runs_resp.status_code == 200:
                runs = runs_resp.json()
                if 'runs' in runs and len(runs['runs']) > 0:
                    latest_run = runs['runs'][0]
                    # Extract metrics safely
                    metrics_list = latest_run['data'].get('metrics', [])
                    metrics = {m['key']: m['value'] for m in metrics_list}
                    return {
                        "run_id": latest_run['info']['run_id'],
                        "accuracy": f"{metrics.get('final_accuracy', 0)*100:.1f}%",
                        "status": "Healthy"
                    }
    except Exception as e:
        print(f"MLflow API Error: {e}")
    return {"run_id": "N/A", "accuracy": "0.0%", "status": "Offline"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def stats():
    # Cassandra Parametrics
    session = get_cassandra_session()
    total_detections = 0
    anomaly_count = 0
    recent_logs = []
    last_audit = "Never"
    
    if session:
        try:
            total_detections = session.execute("SELECT count(*) FROM detections").one()[0]
            anomaly_count = session.execute("SELECT count(*) FROM events WHERE event_type = 'ANOMALY' ALLOW FILTERING").one()[0]
            
            # Get latest timestamp from the optimized table
            last_row = session.execute("SELECT timestamp FROM recent_activity WHERE feed_type='LIVE_FEED' LIMIT 1").one()
            if last_row:
                last_audit = last_row.timestamp.strftime("%H:%M:%S")

            # Get the Live Feed (Unified)
            rows = session.execute("SELECT * FROM recent_activity WHERE feed_type='LIVE_FEED' LIMIT 8")
            for r in rows:
                # Detections usually have 'Confidence' in the details string
                is_anomaly = "Confidence" not in (r.details or "")
                recent_logs.append({
                    "time": r.timestamp.strftime("%H:%M:%S"),
                    "label": r.label if not is_anomaly else f"⚠️ {r.label}",
                    "confidence": r.details
                })
        except Exception as e:
            print(f"Stats fetch error: {e}")
    
    # Check Lockdown Status
    lockdown_status = "ACTIVE"
    if os.path.exists("lockdown_signal.txt"):
        with open("lockdown_signal.txt", "r") as f:
            if f.read().strip() == "1":
                lockdown_status = "LOCKED"

    mlflow = get_mlflow_data()
    
    return jsonify({
        "total_detections": total_detections,
        "anomaly_count": anomaly_count,
        "model_accuracy": mlflow['accuracy'],
        "model_status": mlflow['status'],
        "run_id": mlflow['run_id'][:8], # Short version
        "recent_logs": recent_logs,
        "lockdown_status": lockdown_status,
        "last_audit": last_audit
    })

if __name__ == '__main__':
    app.run(debug=True, port=8080)
