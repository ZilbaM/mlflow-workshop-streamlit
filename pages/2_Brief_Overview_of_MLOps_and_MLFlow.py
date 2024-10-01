# pages/2_Brief_Overview_of_MLOps_and_MLFlow.py

import streamlit as st

st.title("Brief Overview of MLOps and MLFlow")

st.write("""
### What is MLOps?

**MLOps (Machine Learning Operations)** is a set of practices that combines **Machine Learning** and **DevOps**, to deploy and maintain Machine Learning systems in production reliably and efficiently.

#### Key Elements of MLOps:

- **Version Control**: Managing changes to code and data.
- **Continuous Integration/Continuous Deployment (CI/CD)**: Automating the building, testing, and deployment of applications.
- **Monitoring and Logging**: Tracking performance, detecting issues, and ensuring reliability.
- **Experiment Tracking**: Recording experiments to reproduce and compare results.
- **Model Deployment**: Serving models in production environments.

### Where Does MLFlow Fit In?

**MLFlow** is an open-source platform designed to manage the end-to-end machine learning lifecycle. It tackles four primary functions:

1. **MLFlow Tracking**: Recording and querying experiments.
2. **MLFlow Projects**: Packaging ML code in a reusable, reproducible form.
3. **MLFlow Models**: Managing and deploying models from various ML libraries.
4. **Model Registry**: Centralized model store for managing model versions.

In this workshop, we'll focus on **MLFlow Tracking**, which helps with the **Experiment Tracking** aspect of MLOps.
""")
