# src/predict.py
import numpy as np
import pandas as pd
from .model import load_model
from .features import HLAFeatureGenerator, PeptideEmbedder
from typing import Union

class HLAPredictor:
    """
    Main prediction interface for HLA-peptide binding prediction
    """
    def __init__(self, model_path="models/siamese_net.h5"):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained .h5 model file
        """
        self.model = load_model(model_path)
        self.hla_processor = HLAFeatureGenerator()
        self.peptide_embedder = PeptideEmbedder()

    def predict_single(self, peptide: str, hla_allele: str) -> float:
        """
        Predict binding probability for a single peptide-HLA pair
        
        Args:
            peptide: Peptide sequence (8-15 amino acids)
            hla_allele: HLA allele name (e.g. 'HLA-A*02:01')
            
        Returns:
            Binding probability (0-1)
        """
        # Get features
        
        peptide_emb = self.peptide_embedder.embed_peptides([peptide]).values.astype(np.float32)
        hla_feats = np.array([self.hla_processor.get_features(hla_allele)], dtype=np.float32)
        
        # Predict
        return self.model.predict([peptide_emb, hla_feats])[0][0]

    def predict(self, input_file: str) -> pd.DataFrame:
        """
        Batch prediction from input file with automatic type conversion
        
        Args:
            input_file: Path to CSV/Excel file with 'peptide' and 'HLA' columns
          
        Returns:
            DataFrame with added 'prediction_prob' column
        """
        # Read input file
        if input_file.endswith('.csv'):
            input_data = pd.read_csv(input_file)
        elif input_file.endswith(('.xlsx', '.xls')):
            input_data = pd.read_excel(input_file)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")
        if 'HLA_sequence' not in input_data.columns:
            raise ValueError("Input file must contain 'HLA_sequence' column with protein sequences")
        
        # Clean HLA names and ensure proper types
        results = input_data.copy()
        results['HLA'] = results['HLA'].str.replace("*", "", regex=False).astype(str)
        results['peptide'] = results['peptide'].astype(str)
        
        try:

            peptide_embs = self.peptide_embedder.embed_peptides(results['peptide']).values.astype(np.float32)
            hla_feats = np.stack([
              np.array(self.hla_processor.get_features(hla, hla_seq), dtype=np.float32)
              for hla, hla_seq in zip(results['HLA'], results['HLA_sequence'])
              ])
            peptide_embs_df = self.peptide_embedder.embed_peptides(results['peptide'])
            peptide_embs = peptide_embs_df.values.astype(np.float32)
            
            hla_feats_list = []
            for i, (hla, hla_seq) in enumerate(zip(results['HLA'], results['HLA_sequence'])):
                hla_feat = self.hla_processor.get_features(hla, hla_seq)
                hla_feats_list.append(np.array(hla_feat, dtype=np.float32))            
            
            # print("HLA embedding:", hla_feat.shape, "...")
            print("Final dtypes:")
            print(f"Peptide embeddings: {peptide_embs.dtype}")
            print(f"HLA features: {hla_feats.dtype}")
            
            # Predict
            results['prediction_prob'] = self.model.predict([peptide_embs, hla_feats]).flatten()
            return results
            
        except Exception as e:
            print("Prediction failed. Checking data types...")
            print("\nPeptide embeddings dtype:", peptide_embs.dtype if 'peptide_embs' in locals() else 'N/A')
            print("HLA features dtype:", hla_feats.dtype if 'hla_feats' in locals() else 'N/A')
            raise ValueError(f"Prediction error: {str(e)}") from e
