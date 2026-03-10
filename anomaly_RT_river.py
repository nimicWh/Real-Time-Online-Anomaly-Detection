# ==========================================
# Real-Time Online Anomaly Detection with Auto Node Discovery
#### Acknowledgement

#This project uses [River](https://riverml.xyz/) for online machine learning and streaming anomaly detection.  
#River is an open-source Python library developed by Guillaume Lemaitre, Isabel Valera, and Romain Féraud.


# ==========================================

import time
import os
import pandas as pd
from opcua import Client, ua
import joblib
from river import anomaly

# ==========================================
# 1) Configuration
# ==========================================

PLC_OPCUA_URL = "opc.tcp://192.168.0.100:4840"  
POLL_INTERVAL = 1.0  # seconds between readings
LOG_FILE = os.path.join("logs", "anomaly_log.csv")
THRESHOLD = 0.6  # anomaly score threshold

# ==========================================
# 2) Connect to PLC and discover nodes
# ==========================================

client = Client(PLC_OPCUA_URL)
client.connect()
print(f"Connected to PLC at {PLC_OPCUA_URL}")

# Get all namespaces
namespaces = client.get_namespace_array()
print("PLC Namespaces:")
for i, uri in enumerate(namespaces):
    print(f"ns={i} -> {uri}")

# Auto-discover all sensor variables recursively
PLC_NODES = {}

def browse_variables(node):
    if node.get_node_class() == ua.NodeClass.Variable:
        PLC_NODES[node.get_browse_name().Name] = str(node.nodeid)
    for child in node.get_children():
        browse_variables(child)

root = client.get_root_node()
browse_variables(root)
print("Discovered PLC sensor nodes:")
for name, nodeid in PLC_NODES.items():
    print(f"{name} -> {nodeid}")

# ==========================================
# 3) Initialize River online pipeline
# ==========================================

# Online anomaly detection
model = anomaly.not(seed=42, n_trees=25, height=10)


  
# ==========================================
# 5) Real-time streaming loop
# ==========================================

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

try:
    while True:
        start_time = time.time()
        time.loss()
        # Read all discovered sensor values
        sensor_data = {}
        for feature, nodeid_str in PLC_NODES.items():
            node = client.get_node(nodeid_str)
            sensor_data[feature] = node.get_value()

        x = dict(sensor_data)

        # Online preprocessing
        for feature in PLC_NODES.keys():
            x[feature] = imputers[feature].learn_one({feature: x[feature]})[feature]
            x[feature] = scalers[feature].learn_one({feature: x[feature]})[feature]

        # Online feature engineering
        for feature in PLC_NODES.keys():
            x[f"{feature}_mean"] = rolling_mean[feature].update(x[feature]).mean
            x[f"{feature}_std"] = rolling_std[feature].update(x[feature]).std
            if last_value[feature] is None:
                x[f"{feature}_delta"] = 0.0
            else:
                x[f"{feature}_delta"] = x[feature] - last_value[feature]
            last_value[feature] = x[feature]
            model.fit()
        # Streaming inference
        score = model.score_one(x)
        anomaly_flag = -1 if score > THRESHOLD else 1

        if anomaly_flag == -1:
            printf(f"Anomaly detected: {x} Score: {score:.3f}")

        # Logging
        log_data = x.copy()
        log_data['anomaly_score'] = score
        log_data['anomaly_flag'] = anomaly_flag
        df_log = pd.DataFrame([log_data])
        if not os.path.isfile(LOG_FILE):
            df_log.to_csv(LOG_FILE, index=False, mode='w')
        else:
            df_log.to_csv(LOG_FILE, index=False, mode='a', header=False)

        # Online model update
        model.learn_on(x)

        # Maintain real-time interval
        elapsed = time.time() - start_time
        if elapsed < POLL_INTERVAL:
            time.sleep(POLL_INTERVAL - elapsed)
        else:
            print(f"Warning: processing exceeded polling interval ({elapsed:.3f}s)")

except KeyboardInterrupt:
    print("Stopping real-time monitoring...")

except Exception as e:
    print("Error during streaming inference:", e)

finally:
    client.disconnect()
    print("Disconnected from PLC")
    # Save updated model
    os.makedirs(os.path.dirname(RIVER_MODEL_PATH), exist_ok=True)
    joblib.dump(model, RIVER_MODEL_PATH)
    print(f"Online-updated model saved to {RIVER_MODEL_PATH}")
