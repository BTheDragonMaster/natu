"""Substitution matrix setup for alignment"""

from pathlib import Path
from importlib.resources import as_file
from collections import defaultdict
import logging
import re
from typing import Any, Optional

import pandas as pd
import numpy as np
import yaml
from pikachu.general import read_smiles
from pikachu.chem.structure import Structure
from pikachu.fingerprinting.ecfp_4 import ECFP
from pikachu.fingerprinting.similarity import get_jaccard_matrix, get_jaccard_from_ecfp
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
                logger.warning(f"Duplicate variant: {variant.name}")

        else:
            assert variant.name in base_alphabet_set

        added_variants.add(variant.name)

    new_names = [v.name for v in variants_to_add]
    all_names = base_alphabet + new_names  # existing alphabet + all new variants

    def score_name(variant: Variant) -> str:
        """Name of the entry in the original matrix that supplies this variant's base score:
        the variant itself if it is already in the matrix, otherwise the substrate it derives from"""
        if variant.name in base_alphabet_set:
            return variant.name
        assert variant.base is not None
        return variant.base.name

    # Score every pair, including pairs of entries that were already in the base matrix, so that
    # chirality- and N-methylation-aware bonuses are applied consistently across the whole matrix.
    scores = np.empty((len(all_names), len(all_names)), dtype=float)
    for i, row_name in enumerate(all_names):
        row_variant = variant_lookup[row_name]
        for j, col_name in enumerate(all_names):
            col_variant = variant_lookup[col_name]
            base_score = df.loc[score_name(row_variant), score_name(col_variant)]

            chirality_score = 0.0
            methylation_score = 0.0
            if chirality_config["chirality_aware_scoring"]:
                chirality_score = chirality_scores[row_variant.chirality][col_variant.chirality]
            if n_methylation_config["n_methylation_aware_scoring"]:
                methylation_score = n_methylation_scores[row_variant.n_methylation][col_variant.n_methylation]

            scores[i, j] = base_score + chirality_score + methylation_score

    return pd.DataFrame(scores, index=all_names, columns=all_names)


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
                logger.debug(
                    f"Monomers {name_1} and {name_2} in SMILES list are structurally identical. If you do not want to keep both, please remove one of these from your SMILES file")


SIMILARITY_METHODS = ("ecfp", "featmorgan")

PENALTY_MODES = ("interpolate", "subtract")

UNKNOWN_SUBSTRATE_DEFAULTS: dict[str, Any] = {
    "use_closest_structure": False,
    "similarity": "ecfp",
    "min_similarity": "auto",
    "auto_percentile": 95.0,
    "penalty_mode": "interpolate",
    "base_penalty": 0.2,
    "penalty_scale": 5.0,
    "ignore_chirality": False,
}


def strip_stereochemistry(smiles: str) -> str:
    """
    Remove stereochemistry from a SMILES string (tetrahedral '@' and double-bond '/' '\\' markers)

    :param smiles: SMILES string
    :return: SMILES string without stereochemistry
    """
    return re.sub(r"[@/\\]", "", smiles)


def get_featmorgan_generator(include_chirality: bool = True) -> Any:
    """
    Return the RDKit Morgan feature fingerprint generator used for FEATMORGAN similarities

    :param include_chirality: if True, chirality is part of the fingerprint
    :return: RDKit fingerprint generator (radius 2, 2048 bits, feature invariants)
    """
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=include_chirality,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    )


def get_unknown_substrate_settings(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Return the settings for substrates without an equivalent in the PARAS dataset

    :param config: NATU configuration; the 'unknown_substrates' section is read, missing keys take defaults
    :return: dictionary with the keys of UNKNOWN_SUBSTRATE_DEFAULTS
    """
    settings = dict(UNKNOWN_SUBSTRATE_DEFAULTS)
    settings.update((config or {}).get("unknown_substrates") or {})

    if settings["similarity"] not in SIMILARITY_METHODS:
        raise ValueError(f"unknown_substrates.similarity must be one of {SIMILARITY_METHODS}, "
                         f"got {settings['similarity']!r}")
    if settings["penalty_mode"] not in PENALTY_MODES:
        raise ValueError(f"unknown_substrates.penalty_mode must be one of {PENALTY_MODES}, "
                         f"got {settings['penalty_mode']!r}")
    if settings["min_similarity"] != "auto":
        try:
            threshold = float(settings["min_similarity"])
        except (TypeError, ValueError):
            raise ValueError("unknown_substrates.min_similarity must be 'auto' or a number between 0 and 1")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("unknown_substrates.min_similarity must be 'auto' or a number between 0 and 1")
    if not 0.0 <= float(settings["auto_percentile"]) <= 100.0:
        raise ValueError("unknown_substrates.auto_percentile must be between 0 and 100")
    if float(settings["base_penalty"]) < 0 or float(settings["penalty_scale"]) < 0:
        raise ValueError("unknown_substrates.base_penalty and penalty_scale must not be negative")

    return settings


def interpolation_weight(similarity: float, threshold: float) -> float:
    """
    How much a substrate may rely on the scores of its closest PARAS substrate

    0 at or below the similarity threshold (scores stay NATU's average scores), 1 for an identical structure
    (scores are those of the closest PARAS substrate), linear in between.

    :param similarity: similarity to the closest PARAS substrate (0-1)
    :param threshold: similarity threshold (0-1)
    :return: weight between 0 and 1
    """
    if threshold >= 1.0:
        return 1.0 if similarity >= 1.0 else 0.0
    return float(min(1.0, max(0.0, (similarity - threshold) / (1.0 - threshold))))


def closest_structure_penalty(similarity_1: float, similarity_2: float,
                              base_penalty: float, penalty_scale: float) -> float:
    """
    Penalty for two different substrates that are represented by the same PARAS substrate

    A substrate with an equivalent PARAS substrate has similarity 1.0; a substrate that borrows the scores of its
    closest PARAS substrate has the similarity to that substrate. The less similar, the larger the penalty.

    :param similarity_1: similarity of substrate 1 to the shared PARAS substrate (0-1)
    :param similarity_2: similarity of substrate 2 to the shared PARAS substrate (0-1)
    :param base_penalty: fixed part of the penalty
    :param penalty_scale: penalty per unit of dissimilarity
    :return: penalty to subtract from the shared PARAS substrate's self score
    """
    return base_penalty + penalty_scale * ((1.0 - similarity_1) + (1.0 - similarity_2))


def make_fingerprints(name_to_smiles: dict[str, str],
                      name_to_structure: dict[str, Structure],
                      method: str = "ecfp",
                      ignore_chirality: bool = False,
                      names: Optional[list[str]] = None) -> dict[str, Any]:
    """
    Make fingerprints for structural similarity

    :param name_to_smiles: name -> SMILES (used by 'featmorgan', and by 'ecfp' if chirality is ignored)
    :param name_to_structure: name -> PIKAChU structure (used by 'ecfp')
    :param method: 'ecfp' (PIKAChU ECFP4, as for the ECFP matrix) or
        'featmorgan' (RDKit Morgan feature fingerprints, as for the FEATMORGAN matrix)
    :param ignore_chirality: if True, stereochemistry is ignored, e.g. a D-amino acid is identical to its L-form
    :param names: names to fingerprint (default: all names in name_to_smiles)
    :return: name -> fingerprint; names whose SMILES cannot be read are left out
    """
    names = list(name_to_smiles) if names is None else names
    fingerprints: dict[str, Any] = {}

    if method == "ecfp":
        for name in names:
            structure = read_smiles(strip_stereochemistry(name_to_smiles[name])) if ignore_chirality \
                else name_to_structure[name]
            fingerprints[name] = ECFP(structure, iterations=2)
    elif method == "featmorgan":
        generator = get_featmorgan_generator(include_chirality=not ignore_chirality)
        for name in names:
            smiles = strip_stereochemistry(name_to_smiles[name]) if ignore_chirality else name_to_smiles[name]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"RDKit cannot read the SMILES of {name}; it is left out of the similarity search")
                continue
            fingerprints[name] = generator.GetFingerprint(mol)
    else:
        raise ValueError(f"Unknown similarity method {method!r}; choose from {SIMILARITY_METHODS}")

    return fingerprints


def bulk_similarity(query: Any, fingerprints: list[Any], method: str = "ecfp") -> list[float]:
    """
    Similarity (0-1) of one fingerprint to a list of fingerprints

    :param query: fingerprint made by make_fingerprints
    :param fingerprints: fingerprints made by make_fingerprints with the same method
    :param method: 'ecfp' (Jaccard index) or 'featmorgan' (Tanimoto coefficient)
    :return: similarities, in the order of fingerprints
    """
    if method == "ecfp":
        return [1.0 - get_jaccard_from_ecfp(query, fingerprint) for fingerprint in fingerprints]
    if method == "featmorgan":
        return list(DataStructs.BulkTanimotoSimilarity(query, fingerprints))
    raise ValueError(f"Unknown similarity method {method!r}; choose from {SIMILARITY_METHODS}")


def get_similarity_threshold(reference_fingerprints: dict[str, Any], method: str = "ecfp",
                             percentile: float = 95.0) -> float:
    """
    Data-driven similarity threshold: a percentile of the similarities between all pairs of reference substrates

    E.g. percentile 95: a substrate counts as 'close' to a reference substrate only if the two are more similar
    than 95% of all pairs of reference substrates.

    :param reference_fingerprints: reference name -> fingerprint (made by make_fingerprints)
    :param method: 'ecfp' or 'featmorgan'
    :param percentile: percentile (0-100)
    :return: similarity threshold (0-1)
    """
    fingerprints = list(reference_fingerprints.values())
    similarities: list[float] = []
    for i, fingerprint in enumerate(fingerprints[:-1]):
        similarities.extend(bulk_similarity(fingerprint, fingerprints[i + 1:], method))
    if not similarities:
        raise ValueError("At least two reference substrates are needed to compute a similarity threshold")
    return float(np.percentile(similarities, percentile))


def find_closest_substrates(query_names: list[str],
                            name_to_smiles: dict[str, str],
                            name_to_structure: dict[str, Structure],
                            reference_name_to_smiles: dict[str, str],
                            reference_name_to_structure: dict[str, Structure],
                            method: str = "ecfp",
                            ignore_chirality: bool = False,
                            reference_fingerprints: Optional[dict[str, Any]] = None) -> dict[str, tuple[str, float]]:
    """
    For each query substrate, find the structurally most similar reference substrate

    :param query_names: names of the substrates to look up
    :param name_to_smiles: query name -> SMILES
    :param name_to_structure: query name -> PIKAChU structure
    :param reference_name_to_smiles: reference name -> SMILES
    :param reference_name_to_structure: reference name -> PIKAChU structure
    :param method: 'ecfp' (PIKAChU ECFP4, Jaccard index, as for the ECFP matrix) or
        'featmorgan' (RDKit Morgan feature fingerprints, Tanimoto, as for the FEATMORGAN matrix)
    :param ignore_chirality: if True, stereochemistry is ignored, e.g. a D-amino acid is identical to its L-form
    :param reference_fingerprints: precomputed reference fingerprints (same method and chirality setting)
    :return: query name -> (closest reference name, similarity between 0 and 1); ties are broken alphabetically
    """
    if reference_fingerprints is None:
        reference_fingerprints = make_fingerprints(reference_name_to_smiles, reference_name_to_structure,
                                                   method, ignore_chirality)
    reference_names = list(reference_fingerprints)
    fingerprints = [reference_fingerprints[n] for n in reference_names]
    query_fingerprints = make_fingerprints(name_to_smiles, name_to_structure, method, ignore_chirality,
                                           names=list(query_names))

    closest: dict[str, tuple[str, float]] = {}
    for name in query_names:
        if name not in query_fingerprints or not fingerprints:
            continue
        name_to_similarity = dict(zip(reference_names, bulk_similarity(query_fingerprints[name], fingerprints, method)))
        best = max(name_to_similarity.values())
        best_name = min(n for n, sim in name_to_similarity.items() if sim >= best - 1e-9)
        closest[name] = (best_name, float(best))

    return closest


def build_substitution_matrix(smiles_file: Path,
                              matrix_type: SubstitutionMatrix,
                              out_file: Path,
                              config: Optional[dict[str, Any]] = None) -> None:
    """
    Build a substitution matrix for the substrates in a SMILES file and write it to a tab-separated file

    :param smiles_file: tab-separated file with header; substrate name in column 1, SMILES in column 2
    :param matrix_type: type of substitution matrix to build
    :param out_file: output file. For PARAS_BASED matrices, a second file <out_file stem>_paras_mapping.tsv
        records which PARAS substrate each substrate was matched to
    :param config: NATU configuration; if None, the packaged configuration of matrix_type is used.
        Only the 'unknown_substrates' section is used here (PARAS_BASED only)
    """
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
                        logger.debug(
                            f"Substrate {name} matches to multiple SMILES in default dataset: "
                            f"{', '.join(natu_equivalents)}. Choosing the first one. If this is "
                            f"incorrect, manually update the substitution matrix or change the "
                            f"substrate name in your SMILES file to match the name in the default "
                            f"dataset exactly.")
                        name_to_natu_name[name] = natu_equivalents[0]
                else:
                    logger.debug(
                        f"Substrate {name} not found in default dataset. Unable to generate "
                        f"PARAS-based substitution scores for this substrate directly.")

        # Substrates without an equivalent can borrow the scores of their structurally closest PARAS substrate
        if config is None:
            config = yaml.safe_load(matrix_type.get_config().read_text()) or {}
        settings = get_unknown_substrate_settings(config)
        name_to_similarity: dict[str, float] = {name: 1.0 for name in name_to_natu_name}
        name_to_weight: dict[str, float] = {name: 1.0 for name in name_to_natu_name}
        name_to_status: dict[str, str] = {name: "equivalent" for name in name_to_natu_name}
        closest_names: set[str] = set()
        unknown_names = [name for name in name_to_smiles if name not in name_to_natu_name]
        threshold: Optional[float] = None
        interpolate = settings["penalty_mode"] == "interpolate"

        if settings["use_closest_structure"] and unknown_names:
            method = settings["similarity"]
            ignore_chirality = bool(settings["ignore_chirality"])
            reference_fingerprints = make_fingerprints(natu_name_to_smiles, natu_structure_lookup,
                                                       method, ignore_chirality)
            if settings["min_similarity"] == "auto":
                threshold = get_similarity_threshold(reference_fingerprints, method,
                                                     float(settings["auto_percentile"]))
                logger.info(f"Similarity threshold: {threshold:.3f} ({float(settings['auto_percentile']):g}th "
                            f"percentile of {method} similarities between PARAS substrates)")
            else:
                threshold = float(settings["min_similarity"])

            closest = find_closest_substrates(unknown_names, name_to_smiles, structure_lookup,
                                              natu_name_to_smiles, natu_structure_lookup,
                                              method=method, ignore_chirality=ignore_chirality,
                                              reference_fingerprints=reference_fingerprints)
            for name in unknown_names:
                if name not in closest:
                    continue
                natu_name, similarity = closest[name]
                name_to_similarity[name] = similarity
                weight = interpolation_weight(similarity, threshold) if interpolate else 1.0
                above = similarity > threshold if interpolate else similarity >= threshold
                if above and weight > 0.0:
                    name_to_natu_name[name] = natu_name
                    name_to_weight[name] = weight
                    name_to_status[name] = "closest"
                    closest_names.add(name)
                    logger.debug(f"Substrate {name}: using scores of closest PARAS substrate {natu_name} "
                                 f"({method} similarity {similarity:.3f}, weight {weight:.3f})")
                else:
                    logger.debug(f"Substrate {name}: closest PARAS substrate {natu_name} is not above the similarity "
                                 f"threshold ({similarity:.3f} vs {threshold:.3f}); using average scores")

        for name in unknown_names:
            if name not in name_to_natu_name:
                name_to_status[name] = "average"
                logger.debug(f"Substrate {name}: scores default to average scores.")

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
                        score = float(df.loc[natu_name_1, natu_name_2])
                        borrowed = name_1 in closest_names or name_2 in closest_names
                        # two different substrates standing for the same PARAS substrate, at least one of them
                        # only by structural similarity: not identical, so lower the shared self score
                        if name_1 != name_2 and natu_name_1 == natu_name_2 and borrowed:
                            if interpolate:
                                score -= float(settings["base_penalty"])
                            else:
                                score -= closest_structure_penalty(name_to_similarity[name_1],
                                                                   name_to_similarity[name_2],
                                                                   float(settings["base_penalty"]),
                                                                   float(settings["penalty_scale"]))
                        if interpolate and borrowed:
                            # move from NATU's average score (weight 0) to the borrowed score (weight 1)
                            known_1 = name_1 not in closest_names
                            known_2 = name_2 not in closest_names
                            if name_1 == name_2:
                                fallback = avg_self_score
                                weight = name_to_weight[name_1]
                            else:
                                if known_1:
                                    fallback = get_average_non_self_score_from_name(df, natu_name_1)
                                elif known_2:
                                    fallback = get_average_non_self_score_from_name(df, natu_name_2)
                                else:
                                    fallback = avg_non_self_score
                                weight = name_to_weight[name_1] * name_to_weight[name_2]
                            score = fallback + (score - fallback) * weight
                        matrix[name_1][name_2] = score

        matrix_df = pd.DataFrame.from_dict(matrix)

        mapping_file = Path(out_file).with_name(f"{Path(out_file).stem}_paras_mapping.tsv")
        with open(mapping_file, "w", encoding="utf-8") as mapping_out:
            mapping_out.write("substrate\tparas_substrate\tstatus\tsimilarity\tweight\tsimilarity_method\t"
                              "threshold\n")
            for name in names:
                similarity = name_to_similarity.get(name)
                searched = name_to_status[name] != "equivalent" and similarity is not None
                weight = name_to_weight.get(name, 0.0)
                mapping_out.write(f"{name}\t{name_to_natu_name.get(name, '')}\t{name_to_status[name]}\t"
                                  f"{'' if similarity is None else f'{similarity:.4f}'}\t{weight:.4f}\t"
                                  f"{settings['similarity'] if searched else ''}\t"
                                  f"{f'{threshold:.4f}' if searched and threshold is not None else ''}\n")
        logger.info(f"PARAS-based matrix: {sum(s == 'equivalent' for s in name_to_status.values())} equivalent, "
                    f"{len(closest_names)} closest structure, "
                    f"{sum(s == 'average' for s in name_to_status.values())} average scores; see {mapping_file}")

    elif matrix_type == SubstitutionMatrix.ECFP:
        structure_lookup: dict[str, Structure] = {n: read_smiles(name_to_smiles[n]) for n in name_to_smiles.keys()}
        distance_matrix = get_jaccard_matrix(structure_lookup)
        distance_df = pd.DataFrame.from_dict(distance_matrix)
        sim_df = 1 - distance_df
        matrix_df = rescale_similarity_matrix(sim_df)

    elif matrix_type == SubstitutionMatrix.FEATMORGAN:

        structure_lookup: dict[str, Mol] = {n: Chem.MolFromSmiles(name_to_smiles[n]) for n in name_to_smiles.keys()}
        name_to_name_to_feat: dict[str, dict[str, float]] = {}

        generator = get_featmorgan_generator()

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
