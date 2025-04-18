import pandas as pd
import numpy as np
from Bio import pairwise2
from Bio.Align import substitution_matrices
import warnings
warnings.filterwarnings('ignore')

class HLAFeatureGenerator:
    def __init__(self, features_path="data/hla_features.csv"):
        self.feature_columns = [str(i) for i in range(320, 500)]
        self.features = pd.read_csv(features_path)
        self._validate_features()

        for col in self.feature_columns:
            self.features[col] = pd.to_numeric(self.features[col], errors='coerce').astype(np.float32)

        self.substitution_matrix = substitution_matrices.load("BLOSUM62")

        # Create a cache to store precomputed HLA features
        self.hla_features_cache = {}

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

    def get_features(self, hla_allele: str, hla_sequence: str) -> np.ndarray:
        """
        Get features for HLA allele, computing new features if allele is unknown
        """
        # First check the cache for the HLA allele
        if hla_allele in self.hla_features_cache:
            return self.hla_features_cache[hla_allele]

        # If not in the cache, compute the features
        if hla_allele in self.features['HLA'].values:
            features = self.features[self.features['HLA'] == hla_allele][self.feature_columns].values[0].astype(np.float32)
        else:
            print(f"Computing features for new HLA: {hla_allele}")
            weights = self._calculate_similarity_weights(hla_sequence)

            # Calculate weighted average of existing features
            features = np.zeros(180, dtype=np.float32)  # Explicit dtype
            for i, (_, row) in enumerate(self.features.iterrows()):
                feature_values = row[self.feature_columns].values.astype(np.float32)
                features += feature_values * weights[i]

            # Cache the newly computed features
            self.hla_features_cache[hla_allele] = features

        return features

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

    def process_hla_data(self, input_file, output_file="output_with_hla_features.csv"):
        df = pd.read_csv(input_file)

        unique_hla_peptides = df[['HLA', 'HLA_sequence']].drop_duplicates()

        for _, row in unique_hla_peptides.iterrows():
            hla_allele = row['HLA']
            hla_sequence = row['HLA_sequence']
            # Precompute features for unique HLA alleles and cache them
            self.get_features(hla_allele, hla_sequence)

        hla_features_list = []
        for _, row in df.iterrows():
            peptide = row['peptide']
            hlaallele = row['HLA']
            hla_sequence = row['HLA_sequence']

            hla_features = self.get_features(hlaallele, hla_sequence)

            hla_features_list.append({
                'peptide': peptide,
                'HLA': hlaallele,
                'HLA_sequence': hla_sequence,
                'hla_features': hla_features
            })

        hla_features_df = pd.DataFrame(hla_features_list)

        result_df = pd.merge(df, hla_features_df, on=['peptide', 'HLA', 'HLA_sequence'], how='left')

        result_df.to_csv(output_file, index=False)
        print(f"Output saved to {output_file}")

        return result_df
