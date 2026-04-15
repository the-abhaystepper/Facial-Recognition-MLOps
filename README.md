# Face Recognition System - MLOps & Distributed Database Integration

This project has been upgraded with MLOps principles and a distributed database architecture, integrated from the Unit 4 project components.

## New Components

### 1. Distributed Database (Cassandra)
The system now supports storing detection and event logs in a distributed Cassandra database.
- **Docker Container**: A Cassandra 4.0 instance is included in the `docker-compose.yml`.
- **Ingestion**: The `ingest_logs_to_cassandra.py` script migrates data from `detection_log.csv` and `event_log.csv` into Cassandra tables (`detections`, `events`) using PySpark.

### 2. MLOps Integration (MLflow)
Model training is now tracked using MLflow for experimental reproducibility.
- **Tracking**: `train_face_model.py` logs hyperparameters (learning rate, epochs, etc.), metrics (accuracy, loss), and the final `.tflite` model artifact.
- **MLflow UI**: Included in `docker-compose.yml`, accessible at `http://localhost:5000`.

### 3. CI/CD Pipeline
A GitHub Actions workflow (`.github/workflows/face_mlops_pipeline.yml`) automates:
- Dependency installation.
- Automated testing via PyTest.
- Cassandra data ingestion.
- Model training and tracking.

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Java 17 (for PySpark)

### Setup & Run

1. **Start Services**:
   ```bash
   docker-compose up -d
   ```

2. **Ingest Logs to Cassandra**:
   ```bash
   python ingest_logs_to_cassandra.py
   ```

3. **Train Model with MLflow Tracking**:
   ```bash
   python train_face_model.py
   ```

4. **View MLflow Dashboard**:
   Open [http://localhost:5000](http://localhost:5000) in your browser.

## Project Structure
- `docker-compose.yml`: Local infrastructure setup.
- `ingest_logs_to_cassandra.py`: Spark-based ELT pipeline.
- `train_face_model.py`: Training script with MLflow logging.
- `test_app.py`: CI/CD diagnostic tests.
