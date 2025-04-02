# src/predict.py
import numpy as np
import pandas as pd
from .model import load_model
from .features import HLAFeatureGenerator, PeptideEmbedder

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
        peptide_emb = self.peptide_embedder.embed_peptides([peptide]).values
        hla_feats = np.array([self.hla_processor.get_features(hla_allele)])
        
        # Predict
        return self.model.predict([peptide_emb, hla_feats])[0][0]

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
      """
      Batch prediction from DataFrame with automatic type conversion
      
      Args:
          input_data: DataFrame with 'peptide' and 'HLA' columns
          
      Returns:
          DataFrame with added 'prediction_prob' column
      """
    # Create a copy and ensure proper types
    results = input_data.copy()
    results['peptide'] = results['peptide'].astype(str)
    results['HLA'] = results['HLA'].astype(str)
    
    try:
        # Convert features to float32 for TensorFlow
        peptide_embs = self.peptide_embedder.embed_peptides(results['peptide']).values.astype(np.float32)
        hla_feats = np.stack([
        np.array(self.hla_processor.get_features(hla), dtype=np.float32) 
        for hla in results['HLA']
        ])
    
        # Double-check types
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
