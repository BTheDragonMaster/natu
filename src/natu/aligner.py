"""Module for configuring aligner."""

from typing import Literal

from Bio.Align import PairwiseAligner, substitution_matrices


__all__ = ["PairwiseAligner", "substitution_matrices"]


def setup_aligner(
    substitution_matrix: substitution_matrices.Array,
    *,
    mode: Literal["global", "local"] = "global",
    open_internal_gap_score: float = -0.5,
    extend_internal_gap_score: float = -0.1,
    open_end_gap_score: float = -0.2,
    extend_end_gap_score: float = -0.05,
) -> PairwiseAligner:
    """
    Setup a Biopython PairwiseAligner with a substitution matrix and gap scores.

    Internal gaps are gaps inside the alignment and are usually penalized more
    strongly. End gaps are gaps at the left or right side of the alignment and
    are often penalized less strongly when sequences may be incomplete or truncated.

    :param substitution_matrix: Substitution matrix used to score aligned symbols.
    :param mode: Alignmet mode. Must be either "global" or "local".
    :param open_internal_gap_score: Score for opening and internal insertion or deletion gap.
    :param extend_internal_gap_score: Score for extending and internal insertion or deletion gap.
    :param open_end_gap_score: Score for opening a left or right end insertion or deletion gap.
    :param extend_end_gap_score: Score for extending a left or right end insertion or deletion gap.
    :return: Configured Biopython PairwiseAligner instance.
    :raises ValueError: If mode is not "global" or "local".
    """
    if mode not in ("global", "local"):
        raise ValueError(f"mode must be one of 'global' or 'local', got {mode}")

    aligner = PairwiseAligner()
    aligner.mode = mode
    aligner.substitution_matrix = substitution_matrix
    aligner.wildcard = None

    aligner.open_internal_insertion_score = open_internal_gap_score
    aligner.extend_internal_insertion_score = extend_internal_gap_score
    aligner.open_internal_deletion_score = open_internal_gap_score
    aligner.extend_internal_deletion_score = extend_internal_gap_score

    aligner.open_left_insertion_score = open_end_gap_score
    aligner.extend_left_insertion_score = extend_end_gap_score
    aligner.open_right_insertion_score = open_end_gap_score
    aligner.extend_right_insertion_score = extend_end_gap_score

    aligner.open_left_deletion_score = open_end_gap_score
    aligner.extend_left_deletion_score = extend_end_gap_score
    aligner.open_right_deletion_score = open_end_gap_score
    aligner.extend_right_deletion_score = extend_end_gap_score

    return aligner
