import csv
import os
from cassandra.cluster import Cluster
from cassandra.concurrent import execute_concurrent
from datetime import datetime

# Optimized version for larger logs using concurrency
def setup_cassandra_schema(session):
    print("Setting up Cassandra keyspace and tables...")
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS security_system 
        WITH replication = {'class':'SimpleStrategy', 'replication_factor':1};
    """)
    
    session.execute("USE security_system;")
    
    session.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            timestamp timestamp,
            label text,
            confidence float,
            PRIMARY KEY (label, timestamp)
        ) WITH CLUSTERING ORDER BY (timestamp DESC);
    """)
    
    session.execute("""
        CREATE TABLE IF NOT EXISTS events (
            timestamp timestamp,
            event_type text,
            message text,
            PRIMARY KEY (event_type, timestamp)
        ) WITH CLUSTERING ORDER BY (timestamp DESC);
    """)

    session.execute("""
        CREATE TABLE IF NOT EXISTS recent_activity (
            feed_type text,
            timestamp timestamp,
            label text,
            details text,
            PRIMARY KEY (feed_type, timestamp)
        ) WITH CLUSTERING ORDER BY (timestamp DESC);
    """)
    print("Schema setup complete.")

def parse_ts(ts_str):
    ts_str = ts_str.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(ts_str, fmt)
        except:
            continue
    return None

def ingest_detections(session, file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    print(f"Ingesting {file_path} (approx 25k rows)...")
    prepared = session.prepare("""
        INSERT INTO detections (timestamp, label, confidence)
        VALUES (?, ?, ?)
    """)

    with open(file_path, mode='r') as f:
        reader = csv.DictReader(f)
        params = []
        for row in reader:
            ts = parse_ts(row['Timestamp'])
            if ts:
                conf = float(row['Confidence']) if row['Confidence'] and row['Confidence'] != 'None' else 0.0
                params.append((ts, row['Label'], conf))
            
            # Execute in batches for speed and feedback
            if len(params) >= 1000:
                execute_concurrent(session, [(prepared, p) for p in params], raise_on_first_error=False)
                print(f"...processed 1000 rows")
                params = []
        
        # Final batch
        if params:
            execute_concurrent(session, [(prepared, p) for p in params], raise_on_first_error=False)
        
    print(f"Successfully finished {file_path}.")

def ingest_events(session, file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    print(f"Ingesting {file_path}...")
    prepared = session.prepare("""
        INSERT INTO events (timestamp, event_type, message)
        VALUES (?, ?, ?)
    """)

    with open(file_path, mode='r') as f:
        reader = csv.DictReader(f)
        params = []
        for row in reader:
            ts = parse_ts(row['Timestamp'])
            if ts:
                params.append((ts, row['EventType'], row['Message']))
            
            if len(params) >= 1000:
                execute_concurrent(session, [(prepared, p) for p in params], raise_on_first_error=False)
                params = []
        
        if params:
            execute_concurrent(session, [(prepared, p) for p in params], raise_on_first_error=False)
    print(f"Successfully finished {file_path}.")

def main():
    try:
        cluster = Cluster(['127.0.0.1'])
        session = cluster.connect()
        setup_cassandra_schema(session)
        
        ingest_detections(session, 'detection_log.csv')
        ingest_events(session, 'event_log.csv')
        
        cluster.shutdown()
        print("Migration to Distributed Database Finished Successfully.")
        
    except Exception as e:
        print(f"Failed to connect to Cassandra: {e}")

if __name__ == "__main__":
    main()
