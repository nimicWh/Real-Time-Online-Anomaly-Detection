# Real-Time Online Anomaly Detection

This project implements a **real-time streaming anomaly detection pipeline** using **River’s online learning library**. It performs **predictive maintenance** by monitoring sensor data, computing online features, and detecting anomalies on-the-fly.

This project uses for online machine learning (Lemaitre et al., 2023).

---

## Features

- **Online preprocessing**
  - Handles missing values with incremental imputation
  - Optional online scaling for numeric features

- **Online feature engineering**
  - Rolling mean and standard deviation per sensor
  - Delta from previous sample
  - All features computed incrementally for real-time processing

- **Online anomaly detection**
  - Uses River’s `HalfSpaceTrees` (online Isolation Forest)
  - Continuously updates the model with new data (`learn_one`)
  - Computes anomaly scores (`score_one`) in real-time

- **Logging and persistence**
  - Logs anomalies and scores to CSV
  - Saves the online model to disk for warm-starting in future sessions


---

