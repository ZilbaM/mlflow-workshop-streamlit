# pages/6_Running_and_Visualizing_Experiments.py

import streamlit as st

st.title("Running and Visualizing Experiments")

st.write("""
### Step 1: Run Your Training Script

Execute the script:

```bash
python mnist_mlp.py
````
### Step 2: Start the MLFlow UI
In a new terminal window, run:
```bash
mlflow ui
```
The UI will be accessible at http://localhost:5000.

### Step 3: Explore the MLFlow UI

**Experiments Page**
- View all experiments and runs.
- Search and filter runs.

Image Placeholder: Screenshot of the Experiments page.

**Run Details**
- Click on a run to see parameters, metrics, and artifacts.
- Download artifacts like the saved model.

Image Placeholder: Screenshot of Run Details page.

**Compare Runs**
- Select multiple runs to compare their metrics and parameters side by side.
Image Placeholder: Screenshot of comparison view. """)