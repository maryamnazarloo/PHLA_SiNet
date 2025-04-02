# src/features/hla_features.py
import pandas as pd
import numpy as np
from Bio import pairwise2
from Bio.Align import substitution_matrices
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class HLAFeatureGenerator:
    def __init__(self, features_path="data/hla_features.csv"):
        """
        Initialize with precomputed HLA features and sequences
        
        Args:
            features_path: Path to CSV with columns: HLA, HLA_sequence, and 180 feature columns
        """
        self.features = pd.read_csv(features_path)
        self.substitution_matrix = substitution_matrices.load("BLOSUM62")
        self._validate_features()

    def _validate_features(self):
        """Ensure required columns exist"""
        required = ['HLA', 'HLA_sequence'] + [str(i) for i in range(320, 500)]
        missing = set(required) - set(self.features.columns)
        if missing:
            raise ValueError(f"Missing columns in features: {missing}")

    def _calculate_similarity(self, seq1, seq2):
        """Calculate normalized BLOSUM62 similarity score"""
        align = pairwise2.align.globalds(seq1, seq2, 
                                       self.substitution_matrix, -10, -0.5,
                                       one_alignment_only=True)[0]
        max_score = min(len(seq1), len(seq2)) * self.substitution_matrix[('A', 'A')]
        return align.score / max_score

    def get_features(self, hla_allele: str, hla_sequence: str = None) -> np.ndarray:
        """
        Get features for HLA allele, using sequence similarity if allele is new
        
        Args:
            hla_allele: HLA allele name (e.g. 'HLA-A*02:01')
            hla_sequence: Protein sequence (required for new alleles)
            
        Returns:
            numpy array of 180 features
        """
        # Try precomputed features first
        if hla_allele in self.features['HLA'].values:
            return self.features[self.features['HLA'] == hla_allele].iloc[:, 2:].values[0]
            
        # Compute features for new allele
        if not hla_sequence:
            raise ValueError(f"Sequence required for new HLA allele: {hla_allele}")
            
        # Calculate similarity-weighted features
        similarities = []
        for _, row in self.features.iterrows():
            sim = self._calculate_similarity(hla_sequence, row['HLA_sequence'])
            similarities.append(sim)
            
        # Normalize similarities to sum to 1
        similarities = np.array(similarities)
        weights = similarities / similarities.sum()
        
        # Compute weighted average of features
        features = np.average(self.features.iloc[:, 2:], axis=0, weights=weights)
        # Convert to float32 before returning
        features = np.array(features, dtype=np.float32)
        
        # Validation
        if features.dtype != np.float32:
            raise ValueError(f"Features must be float32, got {features.dtype}")
            
        return features

    def save_features(self, output_path: str):
        """Save current features (including any newly computed ones)"""
        self.features.to_csv(output_path, index=False)
