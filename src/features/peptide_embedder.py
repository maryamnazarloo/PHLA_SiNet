# peptide_embedder.py
import torch
import esm
import pandas as pd
import numpy as np
from typing import Union, List, Optional
import logging
import warnings
from tqdm import tqdm

class PeptideEmbedder:
    """
    ESM-2 Peptide Embedder (Original DataFrame Output Version)
    Generates embeddings as a DataFrame with esm_* columns exactly like the first code you shared.
    
    Example:
        embedder = PeptideEmbedder()
        df = embedder.embed_peptides(["ACDEFGHIK", "YKLQPLTFL"])
    """

    MODEL_CONFIG = {
        'esm2_t6_8M': {
            'dim': 320,
            'loader': esm.pretrained.esm2_t6_8M_UR50D
        },
        'esm2_t12_35M': {
            'dim': 480,
            'loader': esm.pretrained.esm2_t12_35M_UR50D
        },
        # ... (other models from your original config) ...
    }

    def __init__(self, model_name: str = 'esm2_t6_8M', device: str = 'auto',
                 batch_size: int = 64, layer: int = 6):
        """Initializes with same parameters as your original code."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.layer = layer
        self.device = self._get_device(device)
        self.model, self.alphabet = self._load_model()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

    def embed_peptides(self, peptides: Union[List[str], pd.Series, pd.DataFrame],
                      peptide_col: Optional[str] = None) -> pd.DataFrame:
        """
        EXACTLY matches your first code's output:
        - Returns DataFrame with esm_* columns
        - Handles all input types (List/Series/DataFrame)
        - Preserves batch processing
        """
        # Convert input to list of sequences
        seq_list = self._prepare_sequences(peptides, peptide_col)
        
        # Process in batches
        all_embeddings = []
        for i in tqdm(range(0, len(seq_list), self.batch_size),
                     desc=f"Embedding ({self.model_name})"):
            batch = seq_list[i:i+self.batch_size]
            batch_embeddings = self._process_batch(batch)
            all_embeddings.append(batch_embeddings)

        return pd.DataFrame(
            np.concatenate(all_embeddings),
            columns=[f"esm_{i}" for i in range(self.embedding_dim)]
        )

    # Helper methods (kept private as in original)
    def _get_device(self, device_str: str) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_str == 'auto' else torch.device(device_str)

    def _load_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, alphabet = self.MODEL_CONFIG[self.model_name]['loader']()
            return model.to(self.device), alphabet

    def _prepare_sequences(self, peptides, peptide_col=None) -> List[str]:
        """Handles all input types exactly like your original code"""
        if isinstance(peptides, pd.DataFrame):
            assert peptide_col is not None, "Must specify peptide_col for DataFrame"
            return peptides[peptide_col].astype(str).tolist()
        elif isinstance(peptides, pd.Series):
            return peptides.astype(str).tolist()
        return list(map(str, peptides))

    def _process_batch(self, batch: List[str]) -> np.ndarray:
        """Batch processing with same embedding logic as original"""
        batch_data = [("", seq) for seq in batch]
        _, _, batch_tokens = self.batch_converter(batch_data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.layer])
            token_representations = results["representations"][self.layer]
            return torch.stack([
                token_representations[i, 1:len(seq)+1].mean(0).cpu()
                for i, seq in enumerate(batch)
            ]).numpy()

    @property
    def embedding_dim(self) -> int:
        return self.MODEL_CONFIG[self.model_name]['dim']

    def save_embeddings(self, peptides: Union[List[str], pd.Series, pd.DataFrame],
                       output_path: str, peptide_col: Optional[str] = None):
        """Original save functionality (CSV/NPY/NPZ)"""
        df = self.embed_peptides(peptides, peptide_col)
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith('.npy'):
            np.save(output_path, df.values)
        elif output_path.endswith('.npz'):
            np.savez(output_path, embeddings=df.values)
        else:
            raise ValueError("Use .csv, .npy, or .npz")
