# BFRB Detection – Artifact Evaluation 1

This repository is a reproduction of the results presented in the paper:  
Zhang, Ryoo, Mukherjee (2025), *Detection of Body Focused Repetitive Behaviors using Deep Learning*.

The artifact is a complete experimental pipeline for the detection of Body Focused Repetitive Behavior (BFRB), which includes data ingestion, exploratory data analysis, data preprocessing, implementation of the model, and evaluation. The entire experimental process is designed as a series of modular Jupyter Notebooks for reproducibility.

---

## System Requirements

The system requirements for the artifact were met during implementation and testing:

- **Operating System**: macOS 14 (tested)
- **CPU**: Apple Silicon (ARM64)
- **RAM**: Minimum 8 GB
- **GPU**: Optional (used only for faster model training)
- **Python**: Version 3.10
- **Key Dependencies**:
  - numpy
  - pandas
  - matplotlib
  - jupyter
  - torch
  - scikit-learn  
  (The full list of dependencies is provided in `requirements.txt`.)

---

## Setup Instructions

To set up the environment and execute the notebooks from a fresh clone of the repository, follow the steps mentioned below.

1. Clone the GitHub repository (≈ 1 minute)
   ```bash
   git clone <repository-url>
   cd 2026-winter-capstone-project-2026winter-capstone-group-5
2. Create and activate a Python virtual environment (optional but recommended) (≈ 2 minutes)
    ```bash
   python3 -m venv venv
   source venv/bin/activate
3. Install all required Python dependencies (≈ 3–5 minutes)
   ```bash
   pip install -r requirements.txt
4. Launch Jupyter Notebook (≈ 1 minute)
   ```bash
   jupyter notebook
5. Open the notebooks/ directory and run the notebooks in the specified order.

---


## Reproducing Results

The experimental process is structured as a series of Jupyter notebooks organized in the notebooks/ directory.
The notebooks must be run in the order specified below to successfully replicate the results.

1. 01_data_ingestion.ipynb

Description: Ingests the raw BFRB data and validates its integrity.
Input: Original dataset files included with the project.
Output: Serialized dataset files stored on disk for use in subsequent notebooks.

2. 02_eda.ipynb

Description: Conducts exploratory data analysis on the ingested dataset.
Tasks:

  Analysis of class distribution
  Summary statistics
  Initial visualizations

Output: EDA visualizations and statistics presented in the notebook.

3. 03_preprocessing.ipynb

Description: Performs data cleaning and preprocessing.
Tasks:

  Missing or noisy data handling
  Feature engineering
  Dataset splitting

Output: Preprocessed datasets stored on disk for model training.

4. 04_models.ipynb

Description: Implements the machine learning and deep learning models as described in the reference paper.
Tasks:

  Model development
  Training and validation
  Output: Trained model artifacts and intermediate results.

5. 05_evaluation.ipynb

Description: Evaluates trained models and replicates the results as reported in the paper.
Metrics:

  Binary F1-score
  Macro-averaged F1-score

Output:
  Evaluation visualizations generated using matplotlib
  Visualizations saved to the plots/ directory.

---


## Limitations

1. The results that are reproduced in this artifact are dependent on the availability of the provided BFRB dataset.
2. The time taken for model training and evaluation may increase for systems without GPU acceleration.
3. Randomly initializing the model parameters may cause differences in the results for the model’s performance metrics.
4. This artifact does not investigate other model types, as the focus is on reproducing the results of the reference paper.
5. The hyperparameters are set to the same values used in the reference study, which may not be optimal for all environments.
   
