# Customer Churn Prediction — End-to-End MLOps & Monitoring Platform

An end-to-end machine learning system for customer churn prediction, covering the complete lifecycle from data processing and model training to production inference, monitoring, and alerting.

The system combines **scikit-learn, MLflow, DAGsHub, FastAPI, Docker, Prometheus, Grafana, Alertmanager, and AWS EC2** to demonstrate a production-oriented MLOps workflow.

## Architecture

<img width="1535" height="1024" alt="ChatGPT Image Aug 23, 2026, 06_43_33 PM" src="https://github.com/user-attachments/assets/7b5f070d-d322-43e1-a48c-4f490c011691" />

### End-to-End Flow

```text
CSV Dataset
    │
    ▼
Data Processing
    │
    ▼
Model Training
    │
    ▼
MLflow / DAGsHub
(Model Registry)
    │
    ▼
FastAPI Inference Service
    │
    ├──────────────► Prometheus
    │                    │
    │                    ├── Node Exporter
    │                    └── DCGM Exporter
    │
    │                    ▼
    │                 Grafana
    │
    │                    ▼
    │              Alertmanager
    │                    │
    │             ┌──────┴──────┐
    │             ▼             ▼
    │           Email        Telegram
    │
    ▼
Prediction
```

## Features

### Machine Learning Pipeline

* Downloaded customer churn dataset from Kaggle
* CSV-based raw and processed datasets
* Data preprocessing and feature engineering using Python
* Model training and evaluation using scikit-learn
* Experiment tracking with MLflow
* Model versioning and lifecycle management through DAGsHub

### Model Registry & Deployment

* Integrated MLflow with DAGsHub as the remote MLflow server
* Managed model versions using aliases:

  * `Champion`
  * `Latest`
* Automatic model loading from the registry
* Champion model used as the preferred production model
* Graceful fallback to the Latest model when the Champion alias is unavailable
* Reproducible model loading in the inference service

### FastAPI Inference Service

* RESTful prediction API built with FastAPI
* Dynamic request schema generation using Pydantic
* Request schema generated from training metadata
* Probability-based churn prediction
* Configurable prediction threshold
* Structured prediction response

### Production Reliability

* Retry mechanism for MLflow service readiness
* Startup dependency checks
* Graceful model loading and alias fallback
* Container health checks
* Dockerized deployment for reproducibility

### Monitoring & Observability

The inference service is instrumented with Prometheus metrics for monitoring both application and model-serving behavior.

#### Application Metrics

* Request count
* Request latency
* Prediction count
* Error rate
* Model inference time

#### System Metrics

* CPU utilization
* Memory utilization
* Disk I/O
* Network I/O
* System load

Collected using **Node Exporter**.

#### GPU Metrics

* GPU utilization
* GPU memory usage
* GPU temperature
* GPU power usage

Collected using **NVIDIA DCGM Exporter** in NVIDIA-compatible environments.

### Monitoring Stack

```text
FastAPI
   │
   │ Prometheus Metrics
   ▼
Prometheus
   │
   ├── Node Exporter
   ├── DCGM Exporter
   │
   ▼
Grafana
   │
   ▼
Alertmanager
   │
   ├── Email
   └── Telegram
```

* Prometheus for metrics collection and time-series storage
* Node Exporter for host-level metrics
* NVIDIA DCGM Exporter for GPU metrics
* Grafana for monitoring dashboards
* Prometheus Alerting Rules for threshold-based alerts
* Alertmanager for alert routing and notification delivery
* Email and Telegram notifications

## Technology Stack

| Category            | Technologies           |
| ------------------- | ---------------------- |
| Language            | Python                 |
| ML                  | scikit-learn, pandas   |
| API                 | FastAPI, Pydantic      |
| Experiment Tracking | MLflow                 |
| Model Registry      | MLflow / DAGsHub       |
| Monitoring          | Prometheus             |
| Dashboards          | Grafana                |
| Host Monitoring     | Node Exporter          |
| GPU Monitoring      | NVIDIA DCGM Exporter   |
| Alerting            | Alertmanager           |
| Notifications       | Email, Telegram        |
| Containerization    | Docker, Docker Compose |
| Deployment          | AWS EC2                |
| Data Format         | CSV                    |

## Example Request

### `POST /predict`

```json
{
  "gender": "Male",
  "tenure": 12,
  "monthly_charges": 70.5
}
```

## Example Response

```json
{
  "prediction": "Yes",
  "probability": 0.82,
  "threshold": 0.5
}
```

## Project Structure

```text
churn_detection/
│
├── data/
│   ├── raw/
│   │   └── *.csv
│   └── processed/
│       └── *.csv
│
├── src/
│   ├── training/
│   ├── inference/
│   └── ...
│
├── app/
│   └── main.py
│
├── monitoring/
│   ├── prometheus/
│   ├── grafana/
│   └── alertmanager/
│
├── models/
│
├── mlruns/
│
├── docker/
│   ├── Dockerfile
│   └── ...
│
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Data Processing

### Download Dataset

The project supports downloading the dataset from Kaggle.

```bash
# Requires Kaggle API credentials
uv run download-data
```

The downloaded dataset is stored as CSV under:

```text
data/raw/
```

### Preprocess Data

```bash
uv run preprocess-data
```

The processed dataset is stored as CSV under:

```text
data/processed/
```

The preprocessing pipeline includes operations such as:

* Data cleaning
* Missing-value handling
* Feature transformation
* Encoding categorical variables
* Feature preparation for model training

## Model Training

Train the churn prediction model and log the experiment to MLflow:

```bash
uv run train-model
```

Training results include:

* Model parameters
* Evaluation metrics
* Model artifacts
* Training metadata
* Model version

The trained model is registered through the MLflow server hosted on **DAGsHub**.

## MLflow / DAGsHub

The project uses DAGsHub as the hosted MLflow server for:

* Experiment tracking
* Parameter logging
* Metric logging
* Artifact tracking
* Model versioning
* Model Registry

### Model Promotion

```text
                 ┌─────────────┐
                 │ New Model   │
                 └──────┬──────┘
                        │
                        ▼
                  Evaluation
                        │
                 ┌──────┴──────┐
                 │             │
              Better?        Worse?
                 │             │
                 ▼             ▼
             Champion       Keep Current
                 │
                 ▼
           Production
```

The inference service attempts to load the `Champion` model first and falls back to `Latest` when the preferred alias is unavailable.

## Docker Deployment

Build and start the complete system:

```bash
docker compose up --build
```

The Docker Compose environment manages the application and monitoring components together.

### Services

| Service          | Purpose                     |
| ---------------- | --------------------------- |
| FastAPI          | ML inference API            |
| Prometheus       | Metrics collection          |
| Grafana          | Monitoring dashboards       |
| Node Exporter    | Host metrics                |
| DCGM Exporter    | NVIDIA GPU metrics          |
| Alertmanager     | Alert routing               |
| MLflow / DAGsHub | Model tracking and registry |

## AWS EC2 Deployment

The system has been deployed on an AWS EC2 instance.

Deployment includes:

* Linux-based EC2 environment
* Docker and Docker Compose
* Security Group configuration
* External API access
* Resource monitoring
* Container resource optimization
* Memory/OOM troubleshooting

Example service endpoints:

```text
FastAPI       → :8000
Grafana       → :3000
Prometheus    → :9090
Alertmanager  → :9093
```

MLflow is hosted remotely through DAGsHub.

## Monitoring & Alerting

Prometheus collects metrics from the application and infrastructure.

Example alerting conditions include:

```text
High CPU Usage
High Memory Usage
High API Latency
High Error Rate
High Model Inference Time
```

Alert flow:

```text
Metric
  │
  ▼
Prometheus
  │
  │ Alerting Rule
  ▼
Alertmanager
  │
  ├──► Email
  │
  └──► Telegram
```

Grafana provides dashboards for:

* API performance
* Request statistics
* Model inference performance
* Host resource utilization
* GPU utilization
* Memory usage
* Monitoring trends over time

## Local Development

Clone the repository:

```bash
git clone https://github.com/yeeislazy/churn_detection.git
cd churn_detection
```

Start the services:

```bash
docker compose up --build
```

Then access the API locally:

```text
http://localhost:8000
```

## Production Demo

The FastAPI inference service has been deployed on **AWS EC2**.

**Live Demo:** Available upon request.

## Key Engineering Highlights

This project demonstrates an end-to-end production-oriented workflow covering:

* Machine learning model development
* Experiment tracking
* Model registry and version management
* Model promotion using aliases
* Dynamic API schema generation
* Containerized deployment
* AWS infrastructure deployment
* Application observability
* Infrastructure monitoring
* GPU monitoring
* Prometheus alerting
* Automated notification delivery
* Production reliability and fault handling

The project is designed to demonstrate practical **MLOps, ML infrastructure, and production ML engineering** rather than model training alone.

