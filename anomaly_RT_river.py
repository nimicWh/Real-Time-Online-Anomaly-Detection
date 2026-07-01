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


# ==========================================
# 2) Connect to PLC and discover nodes
# ==========================================


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
 
