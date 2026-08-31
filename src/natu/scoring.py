from typing import Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np

from natu.aligner import substitution_matrices
from natu.matrix import expand_substitution_matrix, add_wildcard


def create_substitution_matrix(df: pd.DataFrame, config: dict[str, Any], smiles_file: Optional[Path] = None) -> substitution_matrices.Array:
    """
    Create a substitution matrix from a pandas DataFrame.

    :param df: Pandas DataFrame containing the substitution matrix, where the index and columns are the same and represent
        the alphabet, and the values are the scoring values.
    :param config: Dictionary containing NATU configuration
    :param smiles_file: Path to a SMILES file for expanding the substitution matrix with tailoring-aware scoring

    :return: Substitution matrix as a substitution_matrices.Array object.
    """
    tailoring_config = config.get("tailoring", {})

    df = expand_substitution_matrix(df, tailoring_config, smiles_file)
    df = add_wildcard(df, config)

    if df.shape[0] != df.shape[1]:
        raise ValueError(f"substitution matrix must be square (same number of rows and columns), got {df.shape}")

    alphabet = tuple(df.columns)
    data = df.to_numpy(dtype=np.float64)

    sm = substitution_matrices.Array(alphabet, 2, data, np.float64)

    return sm
