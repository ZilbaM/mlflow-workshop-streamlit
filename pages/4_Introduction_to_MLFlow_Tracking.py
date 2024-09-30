
# pages/4_Introduction_to_MLFlow_Tracking.py

import streamlit as st

st.title("Introduction to MLFlow Tracking")

st.write("""
### What is MLFlow Tracking?

MLFlow Tracking is a component that allows you to log and query experiments, including:

- **Parameters**: Hyperparameters used in the model.
- **Metrics**: Performance measurements like accuracy and loss.
- **Artifacts**: Output files like models and plots.
- **Source Code**: The code that ran the experiment.

### Core Concepts

- **Experiment**: A collection of ML runs for a specific task.
- **Run**: A single execution of the training script.

### MLFlow Tracking API

Key functions:

- `mlflow.start_run()`: Starts a new run.
- `mlflow.log_param()`: Logs a single parameter.
- `mlflow.log_params()`: Logs multiple parameters.
- `mlflow.log_metric()`: Logs a single metric.
- `mlflow.log_metrics()`: Logs multiple metrics.
- `mlflow.log_artifact()`: Logs a local file or directory as an artifact.
- `mlflow.set_experiment()`: Sets the active experiment.
""")
