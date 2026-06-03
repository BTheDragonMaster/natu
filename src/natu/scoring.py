"""Scoring functionalities for sequence alignment."""

import numpy as np
import pandas as pd

from natu.aligner import substitution_matrices


def create_substitution_matrix(df: pd.DataFrame) -> substitution_matrices.Array:
    """
    Create a substitution matrix from a pandas DataFrame.

    :param df: Pandas DataFrame containing the substitution matrix, where the index and columns are the same and represent
        the alphabet, and the values are the scoring values.
    :return: Substitution matrix as a substitution_matrices.Array object.
    """
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"substitution matrix must be square (same number of rows and columns), got {df.shape}")

    alphabet = tuple(df.columns)
    data = df.to_numpy(dtype=np.float64)

    sm = substitution_matrices.Array(alphabet, 2, data, np.float64)

    return sm
