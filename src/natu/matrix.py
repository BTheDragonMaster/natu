"""Substitution matrix setup for alignment"""

import numpy as np
import pandas as pd
from typing import Any

from natu.aligner import substitution_matrices
from natu.variants import Modification, Variant, parse_variants
from natu.constants import StructureData, MatrixOptions


def load_matrix_options(matrix_options: MatrixOptions) -> pd.DataFrame:
    """Return dataframe containing scoring matrix from MatrixOptions enum

    :param matrix_options: scoring matrix at natu.data.matrix_options
    :return: pandas dataframe containing the scoring matrix
    """
    with matrix_options.open() as handle:
        scores = pd.read_csv(handle, sep="\t", index_col=0)
        if list(scores.index) != list(scores.columns):
            raise ValueError("Score row names and column names must be identical and in the same order")

    return scores


def get_average_self_score(df: pd.DataFrame) -> float:
    """
    Obtain average substrate self score from matrix diagonal

    :param df: substitution matrix
    :return: average self score
    """

    diagonal = np.diag(df)
    mean_score = diagonal.mean()
    return mean_score


def get_min_scores(df: pd.DataFrame) -> pd.Series:
    """
    Obtain the minimal score per substrate, based on that substrate's row

    :param df: substitution matrix
    :return: Series indexed by substrate, giving the min score per row
    """
    return df.min(axis=1)


def add_wildcard(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Add a wildcard substrate to the matrix

    :param df: substitution matrix
    :param config: Dictionary containing NATU configuration
    :return: expanded matrix, including the wildcard
    """
    matrix_config = config.get("matrix", {})
    if not matrix_config["wildcard_for_unknowns"]:
        return df

    character = matrix_config["wildcard_character"]

    self_score = get_average_self_score(df)
    other_scores = get_min_scores(df)

    if character in df.index or character in df.columns:
        raise ValueError(f"Wildcard character '{character}' already exists in the matrix")

    df = df.copy()

    # 1. add the wildcard column (one new value per existing row)
    df[character] = other_scores

    new_row = pd.concat([other_scores, pd.Series({character: self_score})])
    new_row = new_row.reindex(df.columns)

    assert new_row.isna().sum() == 0, "New row contains missing values before assignment"

    # 2. add the wildcard row (one new value per existing column, including the new one)
    df.loc[character] = new_row

    assert df.isna().sum().sum() == 0, "Matrix contains missing values after adding wildcard"
    assert df.shape[0] == df.shape[1], f"Matrix not square: {df.shape}"

    return df

def expand_substitution_matrix(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Expand substitution matrix with structure variants

    :param df: Pandas DataFrame containing the substitution matrix, where the index and columns are the
        same and represent the alphabet, and the values are the scoring values.
    :param config: Dictionary containing NATU configuration
    :return: Pandas DataFrame containing the expanded substitution matrix
    """
    matrix_config = config.get("matrix", {})

    general_config = config.get("general", {})
    name = general_config.get("name")

    chirality_options = MatrixOptions.from_name(name, "chirality")
    chirality_scores = load_matrix_options(chirality_options)

    methylation_options = MatrixOptions.from_name(name, "methylation")
    methylation_scores = load_matrix_options(methylation_options)

    if matrix_config["chirality_aware_scoring"] and matrix_config["methylation_aware_scoring"]:
        modifications = Modification.D_NME
    elif matrix_config["methylation_aware_scoring"]:
        modifications = Modification.NME
    elif matrix_config["chirality_aware_scoring"]:
        modifications = Modification.D
    else:
        return df

    structure_data = StructureData.from_name(name)

    base_alphabet: list[str] = list(df.columns)
    base_alphabet_set = set(base_alphabet)

    variants_to_add: list[Variant] = []
    all_variants = parse_variants(structure_data)
    variant_lookup = {d.name: d for d in all_variants}

    for variant in all_variants:
        if variant.modification is not None:
            if variant.modification in modifications and variant.name not in base_alphabet_set:
                variants_to_add.append(variant)
        else:
            assert variant.name in base_alphabet_set

    pair_scores: dict[tuple[str, str], float] = {}
    new_names = [v.name for v in variants_to_add]

    all_col_names = base_alphabet + new_names  # existing alphabet + all new variants

    for row_variant in variants_to_add:
        for col_name in all_col_names:
            col_variant = variant_lookup[col_name]
            if col_variant.base is None:
                base_name = col_variant.name
            else:
                base_name = col_variant.base.name
            base_score = df.loc[row_variant.base.name, base_name]

            chirality_score = 0.0
            methylation_score = 0.0
            if Modification.D in modifications:
                chirality_score = chirality_scores.loc[row_variant.chirality.name, col_variant.chirality.name]
            if Modification.NME in modifications:
                methylation_score = methylation_scores.loc[row_variant.methylation.name, col_variant.methylation.name]

            pair_scores[(row_variant.name, col_name)] = base_score + chirality_score + methylation_score

    # --- assemble ---
    def lookup(a: str, b: str) -> float:
        if (a, b) in pair_scores:
            return pair_scores[(a, b)]
        return pair_scores[(b, a)]

    new_cols_df = pd.DataFrame(
        {name: [lookup(name, idx) for idx in df.index] for name in new_names},
        index=df.index,
    )

    new_block = pd.DataFrame(
        {col: [lookup(row, col) for row in new_names] for col in new_names},
        index=new_names,
    )

    df = pd.concat([df, new_cols_df], axis=1)
    new_rows = pd.concat([new_cols_df.T, new_block], axis=1)
    df = pd.concat([df, new_rows.reindex(columns=df.columns)])

    return df


def create_substitution_matrix(df: pd.DataFrame, config: dict[str, Any]) -> substitution_matrices.Array:
    """
    Create a substitution matrix from a pandas DataFrame.

    :param df: Pandas DataFrame containing the substitution matrix, where the index and columns are the same and represent
        the alphabet, and the values are the scoring values.
    :param config: Dictionary containing NATU configuration
    :return: Substitution matrix as a substitution_matrices.Array object.
    """

    df = expand_substitution_matrix(df, config)
    df = add_wildcard(df, config)

    if df.shape[0] != df.shape[1]:
        raise ValueError(f"substitution matrix must be square (same number of rows and columns), got {df.shape}")

    alphabet = tuple(df.columns)
    data = df.to_numpy(dtype=np.float64)

    sm = substitution_matrices.Array(alphabet, 2, data, np.float64)

    return sm
