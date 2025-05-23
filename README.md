# PHLA-SiNet
**Peptide-HLA Interaction Prediction Using Siamese Neural Networks**

![PHLA-SiNet Workflow](docs/workflow.png) 

## Abstract

The Human Leukocyte Antigen (HLA) system is crucial in the immune response, presenting peptides to T cells to distinguish between self and non-self. This study introduces the PHLA-SiNet pipeline, a novel computational approach to predict peptide-HLA (PHLA) interactions, using the diversity and specificity of HLA molecules. We propose an information content-based feature for HLAs, derived from their associated peptides, and utilize ESM embeddings to represent peptides. By employing a Siamese Neural Network (SNN), we predict PHLA interactions, addressing limitations of existing models that rely on HLA names or sequences. Our pipeline enhances prediction accuracy by including a new biological feature for HLA molecules based on binding and non-binding peptides. Additionally, we overcome constraints of models restricted to peptides length by employing a large language model for flexible peptide representation. This approach demonstrates improved performance in predicting PHLA interactions, offering a strong tool for advancing cancer immunotherapy and other HLA-related research.

## Features

- **Novel HLA features**: Information content-based representation derived from binding/non-binding peptides
- **Advanced peptide embeddings**: ESM-2 language model for sequence representation
- **Siamese architecture**: Neural network optimized for interaction prediction
- **Flexible handling**: Works with both known and novel HLA alleles

## Requirements
- Python 3.8+
- TensorFlow 2.6+
- PyTorch 1.12+ (for ESM embeddings)
- fair-esm 0.4.2+
 
## Quick Install
```bash
!git clone https://github.com/maryamnazarloo/PHLA_SiNet.git
%cd PHLA_SiNet
!pip install -r requirements.txt
```
## Usage
### Basic Prediction
```bash
!from hla_predictor import HLAPredictor

# Initialize predictor
predictor = HLAPredictor(
    model_path="models/siamese_net.h5"
)

# Batch prediction from file
results = predictor.predict("input_samples.csv")
```
The easiest way to use PHLA-SiNet is via our Google Colab notebook:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/112QDqAUa_X5_NDLi8sGxI8Q76VL_Eea7?usp=sharing)

Simply:
1. Click the Open in Colab button above
2. Run all cells
3. Upload your input file when prompted
   
### Input File Format
Create a CSV file with these columns (example below):

| HLA        | peptide   | HLA_sequence                  |
|------------|-----------|-------------------------------|
| HLA-A02:01 | ACDEFGHIK | MVVMAPRTLFLL... (pseudo sequence) |
| HLA-B07:02 | YLLPAIVHI | MAVMAPRTLLL...               |

**Note: HLA_sequence is only required for novel HLA alleles not in the training data**

- You can use HLA_pseudo_sequence.dat, which is located in the data section, to find the pseudo sequences for some of HLAs


## Dataset
This project uses training and testing data (both External and Independent) from the TransPHLA-AOMP dataset.  
[![TransPHLA-AOMP Dataset](https://img.shields.io/badge/Dataset-TransPHLA--AOMP-blue?style=flat&logo=github)](https://github.com/a96123155/TransPHLA-AOMP)

## Contact
For any questions, please contact:
Maryam Nazarloo
maryamnazarloo966@gmail.com
