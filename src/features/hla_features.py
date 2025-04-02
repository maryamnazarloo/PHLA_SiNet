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
            features_path: CSV with columns: HLA, HLA_sequence, and 180 feature columns (320-499)
        """
        # Load features and validate structure
        self.features = pd.read_csv(features_path)
        self._validate_features()
        
        # Prepare BLOSUM62 matrix
        self.substitution_matrix = substitution_matrices.load("BLOSUM62")
        
        # Create headers for feature columns (320-499)
        self.feature_columns = [str(i) for i in range(320, 500)]
        
    def _validate_features(self):
        """Ensure required columns exist"""
        required = ['HLA', 'HLA_sequence'] + [str(i) for i in range(320, 500)]
        missing = set(required) - set(self.features.columns)
        if missing:
            raise ValueError(f"Missing columns in features: {missing}")

    def _calculate_similarity(self, seq1, seq2):
        """Calculate normalized BLOSUM62 similarity score between two sequences"""
        align = pairwise2.align.globalds(
            seq1, seq2,
            self.substitution_matrix, -10, -0.5,  # Gap penalties
            one_alignment_only=True
        )[0]
        max_possible = min(len(seq1), len(seq2)) * self.substitution_matrix[('A', 'A')]
        return align.score / max_possible

    def _calculate_similarity_weights(self, new_sequence):
        """
        Calculate similarity weights between new sequence and existing HLA sequences
        Returns: Normalized weights array (sum to 1)
        """
        similarities = []
        for existing_seq in self.features['HLA_sequence']:
            sim = self._calculate_similarity(new_sequence, existing_seq)
            similarities.append(sim)
        
        weights = np.array(similarities)
        return weights / weights.sum()  # Normalize

    def get_features(self, hla_allele: str, hla_sequence: str = None) -> np.ndarray:
        """
        Get features for HLA allele, computing new features if allele is unknown
        
        Args:
            hla_allele: HLA allele name (e.g. 'HLA-A*02:01')
            hla_sequence: Protein sequence (required for new alleles)
            
        Returns:
            numpy array (float32) of 180 features
        """
        # Check for existing features
        if hla_allele in self.features['HLA'].values:
            return self.features[self.features['HLA'] == hla_allele][self.feature_columns].values[0].astype(np.float32)
        
        # Compute features for new allele
        if not hla_sequence:
            raise ValueError(f"Sequence required for new HLA allele: {hla_allele}")
            
        print(f"Computing features for new HLA: {hla_allele}")
        weights = self._calculate_similarity_weights(hla_sequence)
        
        # Calculate weighted average of existing features
        new_features = np.zeros(180)
        for i, (_, row) in enumerate(self.features.iterrows()):
            new_features += row[self.feature_columns].values * weights[i]
        
        # Convert to float32 for TensorFlow
        return new_features.astype(np.float32)

    def add_new_allele(self, hla_allele: str, hla_sequence: str):
        """
        Permanently add a new HLA allele to the feature database
        
        Args:
            hla_allele: HLA allele name
            hla_sequence: Protein sequence
        """
        features = self.get_features(hla_allele, hla_sequence)
        
        new_row = {'HLA': hla_allele, 'HLA_sequence': hla_sequence}
        new_row.update({col: val for col, val in zip(self.feature_columns, features)})
        
        self.features = pd.concat([self.features, pd.DataFrame([new_row])], ignore_index=True)

    def save_features(self, output_path: str):
        """Save current features to CSV"""
        self.features.to_csv(output_path, index=False)
