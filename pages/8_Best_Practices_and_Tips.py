# pages/8_Best_Practices_and_Tips.py

import streamlit as st

st.title("Best Practices and Tips")

st.write("""
### Organizing Experiments

- **Use Descriptive Run Names**

```python
with mlflow.start_run(run_name="LR_0.005_BS_64"):
    # Training code...
```

- **Add Tags for Metadata**

```python
mlflow.set_tag("model", "MLP")
mlflow.set_tag("dataset", "MNIST")
```

### Handling Large Artifacts
- Store models and artifacts remotely if they are large.
- Configure MLFlow to use a remote tracking server or artifact storage.

### Parameterizing Your Script
- Use argparse to make your script more flexible:

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()
```
- Update your hyperparameter logging:

```python
mlflow.log_params(vars(args))
```
""")
