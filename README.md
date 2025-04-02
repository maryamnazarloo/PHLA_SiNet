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
!git clone https://github.com/maryamnazarloo/PHLA_SiNet.git
%cd PHLA_SiNet
!pip install -r requirements.txt
```
# Usage
## Basic Prediction
```bash
!from hla_predictor import HLAPredictor

# Initialize predictor
predictor = HLAPredictor(
    model_path="models/siamese_net.h5",
    hla_feature_path="data/hla_features.csv"
)

# Predict single pair
probability = predictor.predict_single(
    peptide="ACDEFGHIK", 
    hla_allele="HLA-A02:01"
)

# Batch prediction from file
results = predictor.predict("input_samples.csv")
```
## Input File Format
Create a CSV file with these columns (example below):

| HLA        | peptide   | HLA_sequence                  |
|------------|-----------|-------------------------------|
| HLA-A02:01 | ACDEFGHIK | MVVMAPRTLFLL... (pseudo sequence) |
| HLA-B07:02 | YLLPAIVHI | MAVMAPRTLLL...               |
