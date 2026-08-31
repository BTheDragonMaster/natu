"""Substitution matrix setup for alignment"""

from pathlib import Path
from importlib.resources import as_file
from collections import defaultdict
import logging
from typing import Any, Optional

import pandas as pd
import numpy as np
from pikachu.general import read_smiles
from pikachu.chem.structure import Structure
from pikachu.fingerprinting.similarity import get_jaccard_matrix
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from rdkit.Chem.rdchem import Mol

from natu.variants import is_equivalent, parse_smiles
from natu.constants import SubstitutionMatrix, StructureData
from natu.variants import Modification, Variant, get_structure_variants, Chirality, NMethylation


logger = logging.getLogger(__name__)

def get_average_self_score(df: pd.DataFrame) -> float:
    """
    Obtain average substrate self score from matrix diagonal

    :param df: substitution matrix
    :return: average self score
    """

    diagonal = np.diag(df)
    mean_score = diagonal.mean()
    return mean_score


def get_average_non_self_score(df: pd.DataFrame) -> float:
    """
    Obtain average non-self score, excluding the matrix diagonal

    :param df: substitution matrix
    :return: average non-self score
    """
    mask = ~np.eye(len(df), dtype=bool)
    return float(df.to_numpy()[mask].mean())


def get_mean_scores(df: pd.DataFrame) -> pd.Series:
    """
    Obtain the mean score per substrate, based on that substrate's row,
    excluding the diagonal (self-comparison) entry.

    :param df: substitution matrix (square, same row/column labels)
    :return: Series indexed by substrate, giving the mean score per row
        excluding that substrate's diagonal entry.
    """
    mask = np.eye(len(df), dtype=bool)
    return df.where(~mask).mean(axis=1)


def add_wildcard(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Add a wildcard substrate to the matrix

    :param df: substitution matrix
    :param config: Dictionary containing NATU configuration
    :return: expanded matrix, including the wildcard
    """
    wildcard_config = config.get("wildcard", {})
    if not wildcard_config["wildcard_for_unknowns"]:
        return df

    character = wildcard_config["wildcard_character"]

    self_score = get_average_self_score(df)
    other_scores = get_mean_scores(df)

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

def load_tailoring_scores(scoring_config: dict[str, Any], modification: Modification) -> defaultdict[Chirality, dict[Chirality, float]] | defaultdict[NMethylation, dict[NMethylation, float]]:
    """Return tailoring-aware score bonuses and penalties

    :param scoring_config: scoring configuration for chirality or N-methylation
    :param modification: modification for tailoring scores
    :return: dictionary mapping tailoring modification (mis)matches to bit score bonuses/penalties
    """
    if modification == Modification.chirality:
        chirality_scores: defaultdict[Chirality, dict[Chirality, float]] = defaultdict(dict)
        for score_name, score in scoring_config.items():
            name_1, name_2 = score_name.split("_")
            chirality_scores[Chirality[name_1.upper()]][Chirality[name_2.upper()]] = score
            chirality_scores[Chirality[name_2.upper()]][Chirality[name_1.upper()]] = score
        return chirality_scores

    elif modification == Modification.n_methylation:
        nmethylation_scores: defaultdict[NMethylation, dict[NMethylation, float]] = defaultdict(dict)
        for score_name, score in scoring_config.items():
            name_1, name_2 = score_name.split("_")
            nmethylation_scores[NMethylation[name_1.upper()]][NMethylation[name_2.upper()]] = score
            nmethylation_scores[NMethylation[name_2.upper()]][NMethylation[name_1.upper()]] = score
        return nmethylation_scores

    else:
        raise ValueError(f"Unknown modification {modification}")


def expand_substitution_matrix(df: pd.DataFrame, tailoring_config: dict[str, Any], smiles_file: Optional[Path] = None) -> pd.DataFrame:
    """Expand substitution matrix with structure variants

    :param df: Pandas DataFrame containing the substitution matrix, where the index and columns are the
        same and represent the alphabet, and the values are the scoring values.
    :param smiles_file: SMILES file path
    :param tailoring_config: Dictionary containing NATU configuration
    :return: Pandas DataFrame containing the expanded substitution matrix
    """
    chirality_config = tailoring_config.get("chirality", {})
    n_methylation_config = tailoring_config.get("n_methylation", {})

    if not chirality_config or not n_methylation_config:
        logger.warning("WARNING: Tailoring configuration is not set up correctly or completely. Tailoring-aware scoring is disabled.")
        return df

    if smiles_file is None:
        if n_methylation_config["n_methylation_aware_scoring"] or chirality_config["chirality_aware_scoring"]:
            logger.warning("WARNING: No SMILES file provided; cannot expand substitution matrix for tailoring-aware scoring")
        return df

    chirality_scores: defaultdict[Chirality, dict[Chirality, float]] = load_tailoring_scores(
        chirality_config["scoring"], Modification.chirality)

    n_methylation_scores: defaultdict[NMethylation, dict[NMethylation, float]] = load_tailoring_scores(
        n_methylation_config["scoring"], Modification.n_methylation)

    base_alphabet: list[str] = list(df.columns)
    base_alphabet_set = set(base_alphabet)

    variants = get_structure_variants(smiles_file, tailoring_config, base_alphabet)

    variants_to_add: list[Variant] = []
    added_variants: set[str] = set()

    variant_lookup = {d.name: d for d in variants}

    for variant in variants:
        if variant.modifications:
            if variant.name not in base_alphabet_set:
                variants_to_add.append(variant)
            if variant.name in added_variants:
                print("Duplicate variant: ", variant.name)

        else:
            assert variant.name in base_alphabet_set

        added_variants.add(variant.name)

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
            if chirality_config["chirality_aware_scoring"]:
                chirality_score = chirality_scores[row_variant.chirality][col_variant.chirality]
            if n_methylation_config["n_methylation_aware_scoring"]:
                methylation_score = n_methylation_scores[row_variant.n_methylation][col_variant.n_methylation]

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


def get_average_non_self_score_from_name(df: pd.DataFrame, name: str) -> float:
    """
    Obtain the average non-self score for a specific substrate.

    :param df: substitution matrix
    :param name: substrate name (must match a row label)
    :return: average non-self score for the substrate
    """
    row = df.loc[name]
    return float(row[row.index != name].mean())


def write_substitution_matrix(df: pd.DataFrame, out_file: Path) -> None:
    df.to_csv(out_file, sep="\t")


def rescale_similarity_matrix(
    sim_df,
    target_min=-5,
    target_max=7):
    """
    Linearly rescale a chemical similarity matrix into a BLOSUM-style score
    range. Preserves the exact rank order and relative spacing of the input
    similarities -- this is a monotonic affine transform, nothing more.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Symmetric similarity matrix, values in [0, 1] (or any bounded range),
        with matching row and column labels (monomer identifiers).
    target_min, target_max : float
        The output range to map onto, e.g. -2 to 13 to mimic a BLOSUM-style
        spread. The lowest similarity in your data maps to target_min, the
        highest maps to target_max.

    Returns
    -------
    score_df : pandas.DataFrame
        Rescaled substitution matrix, indexed/columned by sim_df's labels.
    info : dict
        Diagnostics: s_min, s_max (the values that got mapped to target_min/
        target_max), scale, offset (the affine transform coefficients).
    """
    assert isinstance(sim_df, pd.DataFrame), "sim_df must be a pandas DataFrame"
    assert list(sim_df.index) == list(sim_df.columns), (
        "sim_df row and column labels must match and be in the same order"
    )

    labels = list(sim_df.index)
    n = len(labels)

    S = sim_df.to_numpy(dtype=float)
    S = (S + S.T) / 2.0  # defensive symmetrization

    s_min = S.min()
    s_max = S.max()

    assert s_max > s_min, "all similarities are identical -- cannot rescale"

    scale = (target_max - target_min) / (s_max - s_min)
    offset = target_min - scale * s_min

    scores = S * scale + offset

    score_df = pd.DataFrame(scores, index=labels, columns=labels)

    return score_df


def check_structures(name_to_structure: dict[str, Structure]) -> None:
    """Check for duplicate names and structures

    :param name_to_structure: dict of [
    :return:
    """
    names = list(name_to_structure.keys())
    names.sort()
    for i, name_1 in enumerate(names):
        structure_1 = name_to_structure[name_1]
        for name_2 in names[i + 1:]:
            structure_2 = name_to_structure[name_2]
            if name_1 == name_2:
                raise ValueError(f"Duplicate substrates found in SMILES input file: {name_1}. Please remove all duplicates.")
            if is_equivalent(structure_1, structure_2):
                logger.warning(
                    f"Monomers {name_1} and {name_2} in SMILES list are structurally identical. If you do not want to keep both, please remove one of these from your SMILES file")


def build_substitution_matrix(smiles_file: Path,
                              matrix_type: SubstitutionMatrix,
                              out_file: Path) -> None:

    name_to_smiles = parse_smiles(smiles_file)
    structure_lookup: dict[str, Structure] = {n: read_smiles(name_to_smiles[n]) for n in name_to_smiles.keys()}

    check_structures(structure_lookup)

    if matrix_type == SubstitutionMatrix.PARAS_BASED:
        matrix: defaultdict[str, dict[str, float]] = defaultdict(dict)
        with as_file(StructureData.PARAS.resource) as path:
            natu_name_to_smiles = parse_smiles(path)
            natu_structure_lookup: dict[str, Structure] = {n: read_smiles(natu_name_to_smiles[n]) for n in natu_name_to_smiles.keys()}

            name_to_natu_name: dict[str, str] = {}
            for name, structure in structure_lookup.items():
                natu_equivalents = []
                for natu_name, natu_structure in natu_structure_lookup.items():
                    if is_equivalent(structure, natu_structure):
                        natu_equivalents.append(natu_name)
                natu_equivalents.sort()

                if len(natu_equivalents) == 1:
                    name_to_natu_name[name] = natu_equivalents[0]
                elif len(natu_equivalents) > 1:
                    if name in natu_equivalents:
                        name_to_natu_name[name] = name
                    else:
                        logger.warning(
                            f"Substrate {name} matches to multiple SMILES in default dataset: "
                            f"{', '.join(natu_equivalents)}. Choosing the first one. If this is "
                            f"incorrect, manually update the substitution matrix or change the "
                            f"substrate name in your SMILES file to match the name in the default "
                            f"dataset exactly.")
                        name_to_natu_name[name] = natu_equivalents[0]
                else:
                    logger.warning(
                        f"Substrate {name} not found in default dataset. Unable to generate "
                        f"PARAS-based substitution scores for this substrate. Scores will default "
                        f"to average scores.")

        with matrix_type.open() as handle:
            df = pd.read_csv(handle, sep="\t", index_col=0)
            if list(df.index) != list(df.columns):
                raise ValueError(
                    "substitution matrix row names and column names must be identical and in the same order")

            names = list(name_to_smiles.keys())

            avg_self_score = get_average_self_score(df)
            avg_non_self_score = get_average_non_self_score(df)

            for name_1 in names:
                natu_name_1 = name_to_natu_name.get(name_1, None)

                for name_2 in names:
                    natu_name_2 = name_to_natu_name.get(name_2, None)

                    if natu_name_1 is None or natu_name_2 is None:
                        if name_1 == name_2:
                            score = avg_self_score
                        elif natu_name_1 is not None:
                            score = get_average_non_self_score_from_name(df, natu_name_1)
                        elif natu_name_2 is not None:
                            score = get_average_non_self_score_from_name(df, natu_name_2)
                        else:
                            score = avg_non_self_score

                        matrix[name_1][name_2] = score
                    else:
                        matrix[name_1][name_2] = float(df.loc[natu_name_1, natu_name_2])

        matrix_df = pd.DataFrame.from_dict(matrix)

    elif matrix_type == SubstitutionMatrix.ECFP:
        structure_lookup: dict[str, Structure] = {n: read_smiles(name_to_smiles[n]) for n in name_to_smiles.keys()}
        distance_matrix = get_jaccard_matrix(structure_lookup)
        distance_df = pd.DataFrame.from_dict(distance_matrix)
        sim_df = 1 - distance_df
        matrix_df = rescale_similarity_matrix(sim_df)

    elif matrix_type == SubstitutionMatrix.FEATMORGAN:

        structure_lookup: dict[str, Mol] = {n: Chem.MolFromSmiles(name_to_smiles[n]) for n in name_to_smiles.keys()}
        name_to_name_to_feat: dict[str, dict[str, float]] = {}

        radius = 2
        n_bits = 2048

        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
            includeChirality=True,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        )

        for name_1, mol_1 in structure_lookup.items():
            fp1 = generator.GetFingerprint(mol_1)
            name_to_name_to_feat[name_1] = {}
            for name_2, mol_2 in structure_lookup.items():
                fp2 = generator.GetFingerprint(mol_2)

                similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

                name_to_name_to_feat[name_1][name_2] = similarity

        feat_df = pd.DataFrame.from_dict(name_to_name_to_feat)
        matrix_df = rescale_similarity_matrix(feat_df)
    elif matrix_type == SubstitutionMatrix.MATCH_MISMATCH:
        matrix: dict[str, dict[str, float]] = {}
        for name_1 in name_to_smiles.keys():
            matrix[name_1] = {}
            for name_2 in name_to_smiles.keys():
                if name_1 == name_2:
                    matrix[name_1][name_2] = 1.0
                else:
                    matrix[name_1][name_2] = 0.0

        matrix_df = pd.DataFrame.from_dict(matrix)
    else:
        raise ValueError(f"Unknown matrix type: {SubstitutionMatrix}")


    matrix_df.to_csv(out_file, sep="\t")
