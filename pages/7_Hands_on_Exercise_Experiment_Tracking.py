# pages/7_Hands_on_Exercise_Experiment_Tracking.py

import streamlit as st

st.title("Hands-on Exercise: Experiment Tracking")

st.write("""
### Objective

Practice tracking experiments by modifying hyperparameters and analyzing results.

### Step 1: Modify Hyperparameters

In `mnist_mlp.py`, change hyperparameters:

```python
learning_rate = 0.005  # Try a different learning rate
batch_size = 64        # Change batch size
```

### Step 2: Run the Experiment
Execute the script again:
```bash
python mnist_mlp.py
```

### Step 3: Repeat with Different Parameters
Experiment with various hyperparameters:

- Learning rates: `0.001`, `0.01`, `0.1`
- Batch sizes: `32`, `64`, `128`

### Step 4: Analyze Results in MLFlow UI
- Use the UI to compare runs.
- Identify which hyperparameters yield the best performance.


""")