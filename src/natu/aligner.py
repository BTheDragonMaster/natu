"""Module for configuring aligner."""

from Bio.Align import PairwiseAligner, substitution_matrices

from natu.constants import AlignmentMode


__all__ = ["PairwiseAligner", "substitution_matrices"]


def setup_aligner(
    substitution_matrix: substitution_matrices.Array,
    *,
    mode: AlignmentMode = AlignmentMode.GLOBAL,
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
    :param mode: Alignment mode. AlignmentMode.GLOCAL (as in BiG-SCAPE: a shorter
        sequence is forced to align in full -- "both arms" -- against a local region
        of a longer one, whose non-matching overhang goes unpenalized) configures a
        global-mode aligner here, but that alone is NOT sufficient to get glocal
        behavior: which side should have its end gaps freed depends on which sequence
        in a given pair is longer, and that varies pair to pair. Every pairwise
        alignment done with a GLOCAL aligner must call
        ``natu.pairwise.configure_glocal_end_gaps`` first, passing that pair's actual
        lengths -- see its docstring. An aligner built with mode=AlignmentMode.GLOCAL
        that never gets reconfigured that way will just behave like plain GLOBAL.
    :param open_internal_gap_score: Score for opening and internal insertion or deletion gap.
    :param extend_internal_gap_score: Score for extending and internal insertion or deletion gap.
    :param open_end_gap_score: Score for opening a left or right end insertion or deletion gap.
    :param extend_end_gap_score: Score for extending a left or right end insertion or deletion gap.
    :return: Configured Biopython PairwiseAligner instance.
    :raises ValueError: If mode is not an AlignmentMode member.
    """
    if not isinstance(mode, AlignmentMode):
        valid = ", ".join(m.name for m in AlignmentMode)
        raise ValueError(f"mode must be one of {valid} (an AlignmentMode member), got {mode!r}")

    aligner = PairwiseAligner()
    # Biopython itself only knows "local"/"global" -- GLOCAL is built on top of "global"
    # by asymmetrically freeing one side's end gaps per pair (see configure_glocal_end_gaps).
    aligner.mode = "global" if mode == AlignmentMode.GLOCAL else mode.value
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
