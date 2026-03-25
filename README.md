# BFRB Detection – Artifact Evaluation 1

This repository is a reproduction of the results presented in the paper:  
Zhang, Ryoo, Mukherjee (2025), *Detection of Body Focused Repetitive Behaviors using Deep Learning*.

The artifact is a complete experimental pipeline for the detection of Body Focused Repetitive Behavior (BFRB), which includes data ingestion, exploratory data analysis, data preprocessing, implementation of the model, and evaluation. The entire experimental process is designed as a series of modular Jupyter Notebooks for reproducibility.The project is structured as a sequence of modular Jupyter notebooks. Each notebook corresponds to a stage in the experimental pipeline, allowing the entire process from raw data to final evaluation to be executed in a clear and reproducible manner.

## Implemented Models

All six models described in the paper have been implemented:
- FFT-MLP (IMU + THM + TOF)
- FFT-MLP (IMU + THM only)
- CNN-BiLSTM (TOF only)
- Late Fusion Ensemble
- Intermediate Fusion Ensemble
- FFT-Random Forest
  
 The architectures, hyperparameters, and training configurations implemented in this artifact strictly adhere to those described in the original   study. No additional models, optimization strategies, hyperparameter tuning procedures, or architectural modifications beyond those reported in   the paper were introduced.

## Notebook Execution Order

To reproduce the full experiment, execute the notebooks in the following order:
- 01_data_ingestion.ipynb
- 02_eda.ipynb
- 03_preprocessing.ipynb
- 04_models.ipynb
- 05_evaluation.ipynb
  
  Each notebook produces outputs that are used by the next stage.
  
## Repository Structure
Initial Structure (After Clone)
After cloning the repository, the structure is:
.
├── data/
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_models.ipynb
│   └── 05_evaluation.ipynb
├── requirements.txt
└── README.md
At this stage, no model artifacts, processed datasets, or evaluation plots are present.
Generated Structure (After Executing Notebooks in Order)

After running the notebooks sequentially, the following directories are created automatically:
.
├── data/
│   └── processed/
│       └── cmi_sensor_data/
│           ├── train_clean.csv
│           ├── train_mlp_scaled.csv
│           ├── val_mlp_scaled.csv
│           └── metadata files
│
├── models_artifacts/
│   ├── models/
│   │   ├── fft_mlp_all_sensors.joblib
│   │   ├── fft_rf_all_sensors.joblib
│   │   ├── cnn_bilstm_tof_model.pt
│   │   └── intermediate_fusion_model.pt
│   │
│   ├── outputs/
│   │   ├── logits_val_fft_mlp_all_sensors.npy
│   │   ├── logits_val_fft_mlp_imu_thm.npy
│   │   ├── logits_val_fft_rf_all_sensors.npy
│   │   ├── validation_logits_cnn_bilstm_tof.npy
│   │   ├── logits_val_late_fusion.npy
│   │   └── validation_logits_intermediate_fusion.npy
│   │
│   └── metadata/
│       ├── class_mapping.json
│       └── split_sequence_ids.json
│
├── plots/
│   └── evaluation/
│       ├── fig4_binary_f1_all_inputs.png
│       ├── fig5_macro_f1_all_inputs.png
│       └── confusion_matrix_best_macro_model.png

All directories under `data/processed/`, `models_artifacts/`, and `plots/`
are generated dynamically during notebook execution and are not required
to be present in a fresh clone of the repository.

## Generated Outputs

When the notebooks are executed in the specified order, the following directories are created automatically:

- models_artifacts/models/  
  Saved trained model weights (.pt, .joblib).

- models_artifacts/outputs/  
  Saved validation logits (.npy) and sequence ID files required for evaluation.

- models_artifacts/metadata/  
  Class mappings and metadata files required for evaluation.

- plots/evaluation/  
  Evaluation plots (Figures 4–7) saved as PNG files.

These files are required for full evaluation reproducibility and are generated dynamically during execution.

## System Requirements

The system requirements for the artifact were met during implementation and testing:

- **Operating System**: macOS 26.2 (tested)
- **CPU**: Apple Silicon (ARM64)
- **RAM**: Minimum 8 GB
- **GPU**: Optional (used only for faster model training)
- **Python**:version tested: 3.14.0  
The artifact may also run on Python ≥3.10.

- **Key Dependencies**:
  - numpy
  - pandas
  - matplotlib
  - jupyter
  - torch
  - scikit-learn  
  (The full list of dependencies is provided in `requirements.txt`.)

## Setup Instructions

To set up the environment and execute the notebooks from a fresh clone of the repository, follow the steps mentioned below.

1. Clone the GitHub repository (≈ 1 minute)
   ```bash
   git clone <repository-url>
   cd 2026-winter-capstone-project-2026winter-capstone-group-5
  Dataset Location: The BFRB dataset files must be placed in the following directory before running the notebooks:
  ```bash
  data/raw/
```
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
- Sequence-aware splitting to prevent data leakage.
- Fixed-length padding/truncation of temporal sequences.
- FFT feature construction for IMU and THM sensors.
- Reshaping of TOF features to (5 × 8 × 8) per time step.
- Standardization of FFT features using training-set statistics only.
- Preservation of validation sequence IDs for reproducible evaluation.

4. 04_models.ipynb

Description: Implements the machine learning and deep learning models as described in the reference paper.
Tasks:

  Model development
  Training and validation
  Output: Trained model artifacts and intermediate results.

5. 05_evaluation.ipynb

Description: Evaluates trained models and replicates the results as reported in the paper.
Metrics:

Metrics:

- Binary F1-score (BFRB vs non-target classification)
- Macro-averaged F1-score across 8 BFRB gesture classes
- Confusion matrix for best macro-F1 model (for analysis)

## Git Branching Strategy

This project follows a feature-branch workflow as required by the artifact rubric:

- Each team member implements their feature in a dedicated branch.
- Feature branches are merged into the `dev` branch via pull requests.
- The `dev` branch is merged into `main` after validation.

This ensures modular development, traceable contributions, and reproducibility.


## Reproducibility Notes

- A fixed random seed (42) is used where applicable.
- Train/test splitting is sequence-aware to prevent data leakage.
- FFT-based models use fixed-length padding/truncation for consistent input dimensions.
- Hyperparameters match those reported in the original paper.
- Minor variations in performance may occur due to hardware or floating-point precision differences.

## Approximate Execution Time

- On a standard laptop (CPU only):
- Data ingestion: < 1 minute
- EDA: 1–2 minutes
- Preprocessing: 2–3 minutes
- Model training: 10–20 minutes
- Evaluation: < 1 minute
  Total reproduction time is approximately 15–25 minutes.

## Limitations

1. The results that are reproduced in this artifact are dependent on the availability of the provided BFRB dataset.
2. The time taken for model training and evaluation may increase for systems without GPU acceleration.
3. Randomly initializing the model parameters may cause differences in the results for the model’s performance metrics.
4. This artifact does not investigate other model types, as the focus is on reproducing the results of the reference paper.
5. The hyperparameters are set to the same values used in the reference study, which may not be optimal for all environments.
6. Reproduction assumes the same class mapping as defined in metadata/class_mapping.json.
7. Evaluation requires all logits files to be generated before running 05_evaluation.ipynb.

# personal-work-college-project
