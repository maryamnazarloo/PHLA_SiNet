# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/112QDqAUa_X5_NDLi8sGxI8Q76VL_Eea7
"""

# hla_predictor/predict.py
import pandas as pd
import numpy as np
import logging
from typing import Union, Optional, Tuple
from pathlib import Path
from .model import load_model
from .features.hla_features import HLAFeatureGenerator
from .features.peptide_embedder import PeptideEmbedder

logger = logging.getLogger(__name__)

class HLAPredictor:
    """
    Main prediction interface for HLA-peptide binding prediction.
    Handles all preprocessing, feature generation, and model prediction.

    Example usage:
        predictor = HLAPredictor(model_path="models/best_model.h5")
        predictions = predictor.predict("input.csv")
    """

    def __init__(self,
                 model_path: str,
                 hla_feature_path: Optional[str] = None,
                 esm_model: str = 'esm2_t6_8M',
                 device: str = 'auto'):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Path to trained .h5 model file
            hla_feature_path: Optional path to precomputed HLA features CSV
            esm_model: ESM model name (default: 'esm2_t6_8M' - 320D)
            device: Computation device ('auto', 'cuda', or 'cpu')
        """
        # Load components
        self.model = load_model(model_path)
        self.embedder = PeptideEmbedder(model_name=esm_model, device=device)

        # Initialize HLA feature generator
        if hla_feature_path:
            self.hla_processor = HLAFeatureGenerator.from_precomputed(hla_feature_path)
        else:
            logger.warning("No HLA feature path provided - only precomputed alleles supported")
            self.hla_processor = HLAFeatureGenerator()

        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Using ESM model: {esm_model}")
        logger.info(f"Running on device: {self.embedder.device}")

    def _prepare_inputs(self,
                       peptides: Union[List[str], np.ndarray],
                       hla_alleles: Union[List[str], np.ndarray],
                       peptide_data: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare model inputs by generating all required features.

        Returns:
            Tuple of (peptide_embeddings, hla_features)
        """
        # Convert inputs to lists if they aren't already
        peptides = list(peptides)
        hla_alleles = list(hla_alleles)

        # Generate peptide embeddings
        logger.info("Generating peptide embeddings...")
        peptide_embeddings = self.embedder.embed_peptides(peptides).values

        # Generate HLA features
        logger.info("Generating HLA features...")
        hla_features = []
        for allele in hla_alleles:
            try:
                features = self.hla_processor.get_features(allele, peptide_data)
                hla_features.append(features)
            except ValueError as e:
                raise ValueError(
                    f"Failed to get features for HLA {allele}. "
                    "For new alleles, provide peptide binding data."
                ) from e

        return peptide_embeddings, np.array(hla_features)

    def predict(self,
                input_data: Union[str, pd.DataFrame, dict],
                peptide_col: str = 'peptide',
                hla_col: str = 'HLA',
                return_probs: bool = True,
                threshold: float = 0.5) -> pd.DataFrame:
        """
        Make predictions for peptide-HLA pairs.

        Args:
            input_data: Can be:
                       - Path to CSV file
                       - Pandas DataFrame
                       - Dictionary {'peptide': [...], 'HLA': [...]}
            peptide_col: Column name for peptide sequences
            hla_col: Column name for HLA alleles
            return_probs: Return probabilities (True) or binary predictions (False)
            threshold: Classification threshold when return_probs=False

        Returns:
            DataFrame with predictions and input data
        """
        # Load and validate input data
        if isinstance(input_data, (str, Path)):
            logger.info(f"Loading input data from {input_data}")
            df = pd.read_csv(input_data)
        elif isinstance(input_data, dict):
            df = pd.DataFrame(input_data)
        else:
            df = input_data.copy()

        # Validate columns
        if peptide_col not in df.columns or hla_col not in df.columns:
            raise ValueError(f"Input data must contain '{peptide_col}' and '{hla_col}' columns")

        # Prepare features
        peptides = df[peptide_col].values
        hla_alleles = df[hla_col].values

        try:
            X_pep, X_hla = self._prepare_inputs(peptides, hla_alleles)
        except Exception as e:
            logger.error("Feature generation failed")
            raise RuntimeError("Could not generate features - check input format") from e

        # Make predictions
        logger.info("Making predictions...")
        preds = self.model.predict([X_pep, X_hla]).flatten()

        # Format results
        results = df.copy()
        if return_probs:
            results['prediction_prob'] = preds
        else:
            results['prediction'] = (preds >= threshold).astype(int)

        logger.info(f"Completed {len(results)} predictions")
        return results

    def predict_single(self,
                      peptide: str,
                      hla_allele: str,
                      return_prob: bool = True,
                      threshold: float = 0.5) -> Union[float, bool]:
        """
        Predict binding for a single peptide-HLA pair.

        Args:
            peptide: Peptide sequence (8-15 amino acids)
            hla_allele: HLA allele (e.g., 'HLA-A*02:01')
            return_prob: Return probability (True) or binary prediction (False)
            threshold: Classification threshold when return_prob=False

        Returns:
            Prediction probability or binary class
        """
        peptide = str(peptide).upper().strip()
        hla_allele = str(hla_allele).strip()

        X_pep, X_hla = self._prepare_inputs([peptide], [hla_allele])
        prob = self.model.predict([X_pep, X_hla]).item()

        return prob if return_prob else (prob >= threshold)

    def save_predictions(self,
                        input_data: Union[str, pd.DataFrame, dict],
                        output_path: str,
                        **predict_kwargs):
        """
        Run predictions and save directly to file.

        Args:
            input_data: See predict() method
            output_path: Output file path (.csv or .tsv)
            predict_kwargs: Additional arguments for predict()
        """
        results = self.predict(input_data, **predict_kwargs)

        if output_path.endswith('.csv'):
            results.to_csv(output_path, index=False)
        elif output_path.endswith('.tsv'):
            results.to_csv(output_path, sep='\t', index=False)
        else:
            raise ValueError("Output format must be .csv or .tsv")

        logger.info(f"Saved predictions to {output_path}")

##example usage:
# Initialize
# predictor = HLAPredictor(
#     model_path="models/best_model.h5",
#     hla_feature_path="data/hla_features.csv"
# )

# # Predict from CSV
# results = predictor.predict("input_samples.csv")

# # Get single prediction
# prob = predictor.predict_single("ACDEFGHIK", "HLA-A*02:01")