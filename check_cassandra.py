from cassandra.cluster import Cluster
try:
    cluster = Cluster(['127.0.0.1'])
    session = cluster.connect('security_system')
    rows = session.execute("SELECT * FROM recent_activity WHERE feed_type='LIVE_FEED' LIMIT 10")
    print("\n--- Cassandra Live Feed (recent_activity) ---")
    for r in rows:
        print(f"Time: {r.timestamp}, Label: {r.label}, Details: {r.details}")
    cluster.shutdown()
except Exception as e:
    print(f"Error: {e}")
