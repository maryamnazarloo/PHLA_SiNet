HLA-peptide-predictor/
│
├── src/                               # All Python source code
│   ├── __init__.py                    # Makes src a Python package
│   ├── predict.py                     # Main prediction interface (HLAPredictor class)
│   ├── model.py                       # Model architecture (build_model function)
│   ├── features/                      # Feature computation
│   │   ├── __init__.py
│   │   ├── hla_features.py            # HLAFeatureGenerator class
│   │   └── peptide_embedder.py        # PeptideEmbedder class
│   └── utils.py                       # Helper functions (if any)
│
├── models/                            # Pretrained models
│   └── best_model.h5                  # Your trained .h5 file
│
├── data/                              # Feature files and sample data
│   ├── hla_features.csv               # Precomputed HLA features
│   └── example_peptides.csv           # Sample input
│
├── notebooks/                         # Jupyter/Colab notebooks
│   ├── Quickstart.ipynb               # Tutorial notebook
│   └── Training_Pipeline.ipynb        # Model training notebook (optional)
│
├── tests/                             # Test scripts
│   ├── test_predict.py
│   └── test_features.py
│
├── docs/                              # Documentation (optional)
│   └── API_Reference.md
│
├── requirements.txt                   # Python dependencies
├── setup.py                           # Installation script
├── LICENSE                            # MIT/Apache/etc.
└── README.md                          # Project documentation

# PHLA_SiNet

A deep learning model for predicting binding interactions between HLA alleles and peptides. This repository provides both the trained model and the tools to run predictions on new data.

## Features

- Predict HLA-peptide binding probabilities
- Handle both known and novel HLA alleles
- Process single sequences or batch files
- Easy Colab integration

## Installation

### Requirements
- Python 3.8+
- TensorFlow 2.6+
- PyTorch (for ESM embeddings)
- fair-esm (ESM-2 models)

### Quick Install
```bash
git clone https://github.com/yourusername/HLA-peptide-predictor.git
cd HLA-peptide-predictor
pip install -r requirements.txt
```
# Usage
## Basic Prediction
```bash
from hla_predictor import HLAPredictor

# Initialize predictor
predictor = HLAPredictor(
    model_path="models/your_model.h5",
    hla_feature_path="data/hla_features.csv"
)

# Predict single pair
probability = predictor.predict_single(
    peptide="ACDEFGHIK", 
    hla_allele="HLA-A*02:01"
)

# Batch prediction from file
results = predictor.predict("input_samples.csv")
```
## Input File Format
Create a CSV file with these columns (example below):

| HLA        | peptide   | HLA_sequence                  |
|------------|-----------|-------------------------------|
| HLA-A*02:01 | ACDEFGHIK | MVVMAPRTLFLL... (pseudo sequence) |
| HLA-B*07:02 | YLLPAIVHI | MAVMAPRTLLL...               |
