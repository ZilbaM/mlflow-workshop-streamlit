# pages/3_Setting_Up_the_Environment.py

import streamlit as st

st.title("Setting Up the Environment")

st.write("""
### Ensure MLFlow is Properly Installed

In this step, we'll make sure that **MLFlow** is installed and accessible via the command-line interface (CLI).

### Step 1: Install MLFlow

Open your terminal and run:

```bash
pip install mlflow
```

*Note: If you're using a virtual environment, ensure it's activated before running the command.*

### Step 2: Verify the Installation
Check that MLFlow is installed correctly by running:

```bash
mlflow --version
```

You should see the MLFlow version printed in the terminal, for example:

```bash
mlflow, version 2.4.1
```

### Step 3: Test MLFlow UI Accessibility

To ensure that the MLFlow UI can be launched, run:

```bash
mlflow ui
```

This should start the MLFlow UI server, accessible at [http://localhost:5000](http://localhost:5000). Open this URL in your web browser to confirm it's working.
**Note**: You can stop the server by pressing `Ctrl+C` in the terminal.

### Conclusion

Now that MLFlow is properly installed and accessible via the CLI, we're ready to integrate it into our PyTorch MNIST MLP code.

""")