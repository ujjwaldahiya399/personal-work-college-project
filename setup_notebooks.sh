#!/usr/bin/env bash
# setup_notebooks.sh
# Creates the 3 NLP-pipeline notebook skeletons in notebooks/
# and ensures plots/ is tracked by Git.
# Run from the root of the repository.

set -e

NOTEBOOKS_DIR="$(dirname "$0")/notebooks"
PLOTS_DIR="$(dirname "$0")/plots"

mkdir -p "$NOTEBOOKS_DIR"
mkdir -p "$PLOTS_DIR"
touch "$PLOTS_DIR/.gitkeep"

echo "Writing notebooks/nlp_model.ipynb ..."
python3 - <<'PYEOF'
import json, os

nb = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "cells": [
  {"cell_type":"markdown","metadata":{},"source":"# NLP-Based BFRB Detection Model\n\nThis notebook builds and trains an NLP-based model for Body-Focused Repetitive Behavior (BFRB) detection.\nIt processes sensor data features as sequential text-like representations and applies NLP techniques for classification."},
  {"cell_type":"markdown","metadata":{},"source":"## 1. Imports and Configuration"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
DATA_PATH = '../data/processed/cmi_sensor_data/train_mlp_scaled.csv'
VAL_DATA_PATH = '../data/processed/cmi_sensor_data/val_mlp_scaled.csv'
METADATA_PATH = '../models_artifacts/metadata/class_mapping.json'
OUTPUT_DIR = '../models_artifacts/outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)"""},
  {"cell_type":"markdown","metadata":{},"source":"## 2. Load Preprocessed Data"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""train_df = pd.read_csv(DATA_PATH)
val_df = pd.read_csv(VAL_DATA_PATH)

print(f'Train shape: {train_df.shape}')
print(f'Val shape:   {val_df.shape}')
train_df.head()"""},
  {"cell_type":"markdown","metadata":{},"source":"## 3. Feature Extraction and Label Encoding"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""# Load class mapping
with open(METADATA_PATH, 'r') as f:
    class_mapping = json.load(f)

label_col = 'label'

X_train = train_df.drop(columns=[label_col]).values
y_train = train_df[label_col].values

X_val = val_df.drop(columns=[label_col]).values
y_val = val_df[label_col].values

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)

print(f'Classes: {le.classes_}')
print(f'X_train shape: {X_train.shape}, y_train shape: {y_train_enc.shape}')"""},
  {"cell_type":"markdown","metadata":{},"source":"## 4. NLP-Inspired Feature Representation\n\nSensor time-series windows are tokenized into discrete bins (analogous to words) using quantile-based binning. Each sequence of bins forms a \"sentence\" that is passed to a TF-IDF-style feature extractor."},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""from sklearn.feature_extraction.text import TfidfVectorizer

N_BINS = 50

def tokenize_sensor_row(row, n_bins=N_BINS):
    \"\"\"Convert a numeric feature vector into a sequence of bin tokens.\"\"\"
    bins = np.linspace(row.min(), row.max() + 1e-9, n_bins + 1)
    token_ids = np.digitize(row, bins) - 1
    return ' '.join([f't{i}b{b}' for i, b in enumerate(token_ids)])

train_docs = [tokenize_sensor_row(row) for row in X_train]
val_docs   = [tokenize_sensor_row(row) for row in X_val]

print(f'Sample token sequence (first 80 chars): {train_docs[0][:80]}...')"""},
  {"cell_type":"markdown","metadata":{},"source":"## 5. TF-IDF Vectorization"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(train_docs)
X_val_tfidf   = vectorizer.transform(val_docs)

print(f'TF-IDF train matrix: {X_train_tfidf.shape}')
print(f'TF-IDF val matrix:   {X_val_tfidf.shape}')"""},
  {"cell_type":"markdown","metadata":{},"source":"## 6. Model Training \u2014 Logistic Regression (NLP Baseline)"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""from sklearn.linear_model import LogisticRegression

nlp_model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=RANDOM_SEED,
    solver='lbfgs',
    multi_class='multinomial'
)

nlp_model.fit(X_train_tfidf, y_train_enc)
print('Model training complete.')"""},
  {"cell_type":"markdown","metadata":{},"source":"## 7. Validation Inference and Logits Export"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""y_pred_enc = nlp_model.predict(X_val_tfidf)
logits_val = nlp_model.predict_proba(X_val_tfidf)

binary_f1 = f1_score(y_val_enc, y_pred_enc, average='binary', pos_label=1)
macro_f1  = f1_score(y_val_enc, y_pred_enc, average='macro')

print(f'Validation Binary F1:  {binary_f1:.4f}')
print(f'Validation Macro F1:   {macro_f1:.4f}')
print()
print(classification_report(y_val_enc, y_pred_enc, target_names=le.classes_))

# Save logits and results for evaluation notebook
np.save(os.path.join(OUTPUT_DIR, 'logits_val_nlp_model.npy'), logits_val)

results = {
    'model': 'NLP-TF-IDF-LogisticRegression',
    'binary_f1': binary_f1,
    'macro_f1': macro_f1,
    'classes': list(le.classes_)
}

with open(os.path.join(OUTPUT_DIR, 'nlp_model_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print('\\nLogits and results saved to models_artifacts/outputs/')"""},
  {"cell_type":"markdown","metadata":{},"source":"## 8. Summary"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""print('=== NLP Model Summary ===')
print(f\"Model      : {results['model']}\")
print(f\"Binary F1  : {results['binary_f1']:.4f}\")
print(f\"Macro F1   : {results['macro_f1']:.4f}\")
print()
print('Outputs written:')
print('  models_artifacts/outputs/logits_val_nlp_model.npy')
print('  models_artifacts/outputs/nlp_model_results.json')"""}
 ]
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.', 'notebooks', 'nlp_model.ipynb')
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'  -> Written: {out}')
PYEOF

echo "Writing notebooks/evaluation.ipynb ..."
python3 - <<'PYEOF'
import json, os

nb = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "cells": [
  {"cell_type":"markdown","metadata":{},"source":"# NLP Model Evaluation vs Baseline Paper Results\n\nThis notebook evaluates the NLP-based BFRB detection model against the baseline results reported in:\n> Zhang, Ryoo, Mukherjee (2025), *Detection of Body Focused Repetitive Behaviors using Deep Learning*.\n\n**Metrics used:**\n- Binary F1-score (BFRB vs non-target)\n- Macro-averaged F1-score across 8 BFRB gesture classes"},
  {"cell_type":"markdown","metadata":{},"source":"## 1. Imports and Configuration"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

# Paths — outputs written by nlp_model.ipynb
OUTPUTS_DIR = '../models_artifacts/outputs/'
METADATA_PATH = '../models_artifacts/metadata/class_mapping.json'
NLP_RESULTS_PATH = os.path.join(OUTPUTS_DIR, 'nlp_model_results.json')
NLP_LOGITS_PATH  = os.path.join(OUTPUTS_DIR, 'logits_val_nlp_model.npy')
PLOTS_DIR = '../plots/'
os.makedirs(PLOTS_DIR, exist_ok=True)"""},
  {"cell_type":"markdown","metadata":{},"source":"## 2. Load NLP Model Results\n\n> **Prerequisite:** Run `nlp_model.ipynb` first to generate `nlp_model_results.json` and `logits_val_nlp_model.npy`."},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""with open(NLP_RESULTS_PATH, 'r') as f:
    nlp_results = json.load(f)

nlp_logits = np.load(NLP_LOGITS_PATH)

print('NLP Model Results:')
print(json.dumps(nlp_results, indent=2))
print(f'\\nLogits shape: {nlp_logits.shape}')"""},
  {"cell_type":"markdown","metadata":{},"source":"## 3. Baseline Paper Results (Table from Zhang et al., 2025)"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""# Results as reported in Zhang, Ryoo, Mukherjee (2025)
baseline_results = {
    'FFT-MLP (IMU+THM+TOF)': {'binary_f1': 0.97, 'macro_f1': 0.73},
    'FFT-MLP (IMU+THM)':     {'binary_f1': 0.95, 'macro_f1': 0.68},
    'CNN-BiLSTM (TOF)':      {'binary_f1': 0.91, 'macro_f1': 0.61},
    'Late Fusion Ensemble':  {'binary_f1': 0.97, 'macro_f1': 0.75},
    'Intermediate Fusion':   {'binary_f1': 0.96, 'macro_f1': 0.72},
    'FFT-Random Forest':     {'binary_f1': 0.94, 'macro_f1': 0.65},
}

baseline_df = pd.DataFrame(baseline_results).T
baseline_df.index.name = 'Model'
print('Baseline Paper Results:')
baseline_df"""},
  {"cell_type":"markdown","metadata":{},"source":"## 4. Comparison Table: NLP Model vs Baselines"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""comparison = baseline_df.copy()
comparison.loc['NLP-TF-IDF-LR (Ours)'] = [
    nlp_results['binary_f1'],
    nlp_results['macro_f1']
]

comparison = comparison.sort_values('macro_f1', ascending=False)
comparison.columns = ['Binary F1', 'Macro F1']
comparison = comparison.round(4)

print('Full Comparison Table:')
comparison"""},
  {"cell_type":"markdown","metadata":{},"source":"## 5. Binary F1 Comparison Bar Chart"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""fig, ax = plt.subplots(figsize=(10, 5))
colors = ['steelblue' if idx != 'NLP-TF-IDF-LR (Ours)' else 'darkorange'
          for idx in comparison.index]

comparison['Binary F1'].plot(kind='barh', ax=ax, color=colors)
ax.set_xlabel('Binary F1-Score')
ax.set_title('Binary F1-Score: NLP Model vs Paper Baselines')
ax.axvline(x=nlp_results['binary_f1'], color='darkorange', linestyle='--', linewidth=1.2, label='NLP model')
ax.legend()
ax.set_xlim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eval_binary_f1_comparison.png'), dpi=150)
plt.show()
print('Saved: plots/eval_binary_f1_comparison.png')"""},
  {"cell_type":"markdown","metadata":{},"source":"## 6. Macro F1 Comparison Bar Chart"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""fig, ax = plt.subplots(figsize=(10, 5))
comparison['Macro F1'].plot(kind='barh', ax=ax, color=colors)
ax.set_xlabel('Macro-Averaged F1-Score')
ax.set_title('Macro F1-Score: NLP Model vs Paper Baselines')
ax.axvline(x=nlp_results['macro_f1'], color='darkorange', linestyle='--', linewidth=1.2, label='NLP model')
ax.legend()
ax.set_xlim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eval_macro_f1_comparison.png'), dpi=150)
plt.show()
print('Saved: plots/eval_macro_f1_comparison.png')"""},
  {"cell_type":"markdown","metadata":{},"source":"## 7. Save Comparison Table to CSV"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""comparison_out_path = os.path.join(OUTPUTS_DIR, 'evaluation_comparison.csv')
comparison.to_csv(comparison_out_path)
print(f'Comparison table saved to: {comparison_out_path}')
print('\\n--- This file is the input for plot_generation.ipynb ---')"""},
  {"cell_type":"markdown","metadata":{},"source":"## 8. Summary"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""best_baseline_macro  = baseline_df['macro_f1'].max()
best_baseline_binary = baseline_df['binary_f1'].max()

print('=== Evaluation Summary ===')
print(f\"NLP Model Binary F1 : {nlp_results['binary_f1']:.4f}  (best baseline: {best_baseline_binary:.4f})\")
print(f\"NLP Model Macro F1  : {nlp_results['macro_f1']:.4f}  (best baseline: {best_baseline_macro:.4f})\")
delta_macro = nlp_results['macro_f1'] - best_baseline_macro
print(f\"\\nMacro F1 gap vs best baseline: {delta_macro:+.4f}\")"""}
 ]
}

out = os.path.join('notebooks', 'evaluation.ipynb')
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'  -> Written: {out}')
PYEOF

echo "Writing notebooks/plot_generation.ipynb ..."
python3 - <<'PYEOF'
import json, os

nb = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "cells": [
  {"cell_type":"markdown","metadata":{},"source":"# Plot Generation\n\nThis notebook reads evaluation results from `evaluation.ipynb` and generates publication-quality plots saved to the `plots/` directory.\n\n**Plots generated:**\n1. Binary F1-score comparison (all models)\n2. Macro F1-score comparison (all models)\n3. Delta / gap chart: NLP model vs each baseline\n\n> **Prerequisite:** Run `nlp_model.ipynb` then `evaluation.ipynb` first to generate `models_artifacts/outputs/evaluation_comparison.csv`."},
  {"cell_type":"markdown","metadata":{},"source":"## 1. Imports and Configuration"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

# Paths — input files written by evaluation.ipynb
COMPARISON_CSV   = '../models_artifacts/outputs/evaluation_comparison.csv'
NLP_RESULTS_JSON = '../models_artifacts/outputs/nlp_model_results.json'
PLOTS_DIR = '../plots/'
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})"""},
  {"cell_type":"markdown","metadata":{},"source":"## 2. Load Evaluation Data"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""comparison = pd.read_csv(COMPARISON_CSV, index_col=0)

with open(NLP_RESULTS_JSON, 'r') as f:
    nlp_results = json.load(f)

NLP_MODEL_LABEL = 'NLP-TF-IDF-LR (Ours)'

print('Loaded comparison table:')
comparison"""},
  {"cell_type":"markdown","metadata":{},"source":"## 3. Plot 1 \u2014 Binary F1-Score Comparison"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""fig, ax = plt.subplots(figsize=(10, 5))

colors = ['darkorange' if idx == NLP_MODEL_LABEL else 'steelblue'
          for idx in comparison.index]

bars = ax.barh(comparison.index, comparison['Binary F1'], color=colors, edgecolor='white')

for bar, val in zip(bars, comparison['Binary F1']):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=10)

nlp_patch  = mpatches.Patch(color='darkorange', label='NLP Model (Ours)')
base_patch = mpatches.Patch(color='steelblue',  label='Paper Baselines')
ax.legend(handles=[nlp_patch, base_patch], loc='lower right')

ax.set_xlabel('Binary F1-Score')
ax.set_title('Figure 1 \u2014 Binary F1-Score: NLP Model vs Paper Baselines')
ax.set_xlim(0, 1.12)
plt.tight_layout()

out_path = os.path.join(PLOTS_DIR, 'fig1_binary_f1_nlp_vs_baselines.png')
plt.savefig(out_path, bbox_inches='tight')
plt.show()
print(f'Saved: {out_path}')"""},
  {"cell_type":"markdown","metadata":{},"source":"## 4. Plot 2 \u2014 Macro F1-Score Comparison"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.barh(comparison.index, comparison['Macro F1'], color=colors, edgecolor='white')

for bar, val in zip(bars, comparison['Macro F1']):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=10)

ax.legend(handles=[nlp_patch, base_patch], loc='lower right')
ax.set_xlabel('Macro-Averaged F1-Score')
ax.set_title('Figure 2 \u2014 Macro F1-Score: NLP Model vs Paper Baselines')
ax.set_xlim(0, 1.12)
plt.tight_layout()

out_path = os.path.join(PLOTS_DIR, 'fig2_macro_f1_nlp_vs_baselines.png')
plt.savefig(out_path, bbox_inches='tight')
plt.show()
print(f'Saved: {out_path}')"""},
  {"cell_type":"markdown","metadata":{},"source":"## 5. Plot 3 \u2014 Performance Gap: NLP Model vs Each Baseline"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""baselines_only = comparison.drop(index=NLP_MODEL_LABEL)

delta_binary = nlp_results['binary_f1'] - baselines_only['Binary F1']
delta_macro  = nlp_results['macro_f1']  - baselines_only['Macro F1']

delta_df = pd.DataFrame({'Binary F1 Delta': delta_binary, 'Macro F1 Delta': delta_macro})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, title in zip(axes,
                           ['Binary F1 Delta', 'Macro F1 Delta'],
                           ['Binary F1 Gap', 'Macro F1 Gap']):
    bar_colors = ['forestgreen' if v >= 0 else 'crimson' for v in delta_df[col]]
    bars = ax.barh(delta_df.index, delta_df[col], color=bar_colors, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    for bar, val in zip(bars, delta_df[col]):
        offset = 0.003 if val >= 0 else -0.003
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}', va='center', fontsize=9)
    ax.set_xlabel('NLP model score \u2212 baseline score')
    ax.set_title(f'Figure 3 \u2014 {title}: NLP Model vs Each Baseline')

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, 'fig3_performance_gap_nlp_vs_baselines.png')
plt.savefig(out_path, bbox_inches='tight')
plt.show()
print(f'Saved: {out_path}')"""},
  {"cell_type":"markdown","metadata":{},"source":"## 6. Summary of Generated Plots"},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":
"""plot_files = sorted([
    f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')
])

print('=== Plots saved to plots/ ===')
for f in plot_files:
    print(f'  {f}')"""}
 ]
}

out = os.path.join('notebooks', 'plot_generation.ipynb')
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'  -> Written: {out}')
PYEOF

echo ""
echo "Done! The following files were created:"
echo "  notebooks/nlp_model.ipynb"
echo "  notebooks/evaluation.ipynb"
echo "  notebooks/plot_generation.ipynb"
echo "  plots/.gitkeep"

