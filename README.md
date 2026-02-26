# Python Project Template

## Workflow Guide

### 1. Initial Setup

First, clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd Python_Project_Template
```

### 2. Virtual Environment Setup

Create and activate a virtual environment:

```bash
# Create virtual environment
make venv

# Activate virtual environment (shown after running make venv)
source venv/bin/activate  # For Unix/MacOS
# or
.\venv\Scripts\activate  # For Windows
```

### 3. Install Dependencies

Install required packages:

```bash
make install
```

### 4. GitHub Integration

Initialize repository and push to GitHub:

```bash
# Set your GitHub username and repo name in Makefile
# Then run:
make init

# For subsequent pushes:
make push
```

### 5. Sync with GitHub

To pull latest changes and update packages:

```bash
make sync
```

### 6. Sample Analysis

The project includes a raw dataset (`raw_data/boston.txt`) and configuration (`src/config/config.yaml`) for running a random forest regression. The configuration file allows you to adjust model hyperparameters without changing the code.

To run the analysis:

1. Activate your virtual environment.
2. Preprocess the raw data to prepare the dataset:
   ```bash
   python src/00_rawdata_process.py
   ```
3. Execute the regression analysis script:
   ```bash
   python src/01_regression.py
   ```
4. Visualize the results using the visualization script:
   ```bash
   python src/02_visualization.py
   ```

Example configuration in `config.yaml`:
```yaml
model:
  type: "random_forest"
  parameters:
    n_estimators: 100
    max_depth: 10
  test_size: 0.2
  scoring: "roc_auc"
```

