"""Tests for natu.aligner.setup_aligner.

Layout: one test class per function under test, named Test<FunctionName>,
with each case as a method on that class.
"""

from __future__ import annotations

import pytest

from natu.aligner import setup_aligner
from natu.constants import AlignmentMode


class TestSetupAligner:
    def test_rejects_invalid_mode(self, substitution_matrix):
        with pytest.raises(ValueError, match="mode must be one of"):
            setup_aligner(substitution_matrix, mode="banana")

    @pytest.mark.parametrize("mode", list(AlignmentMode))
    def test_accepts_valid_modes(self, substitution_matrix, mode):
        aligner = setup_aligner(substitution_matrix, mode=mode)
        assert aligner is not None

    def test_glocal_is_implemented_as_global_mode(self, substitution_matrix):
        """
        Biopython itself only knows "local"/"global" -- setup_aligner's own
        docstring says AlignmentMode.GLOCAL configures a global-mode aligner
        that must be reconfigured per-pair (via configure_glocal_end_gaps)
        to behave differently. An aligner built with mode=AlignmentMode.GLOCAL
        that's never reconfigured should behave exactly like plain GLOBAL.
        """
        aligner = setup_aligner(substitution_matrix, mode=AlignmentMode.GLOCAL)
        assert aligner.mode == "global"

    def test_wires_internal_and_end_gap_scores(self, substitution_matrix):
        aligner = setup_aligner(
            substitution_matrix,
            mode=AlignmentMode.GLOBAL,
            open_internal_gap_score=-9.0,
            extend_internal_gap_score=-8.0,
            open_end_gap_score=-7.0,
            extend_end_gap_score=-6.0,
        )

        assert aligner.open_internal_insertion_score == -9.0
        assert aligner.extend_internal_insertion_score == -8.0
        assert aligner.open_internal_deletion_score == -9.0
        assert aligner.extend_internal_deletion_score == -8.0

        assert aligner.open_left_insertion_score == -7.0
        assert aligner.extend_left_insertion_score == -6.0
        assert aligner.open_right_insertion_score == -7.0
        assert aligner.extend_right_insertion_score == -6.0
        assert aligner.open_left_deletion_score == -7.0
        assert aligner.extend_left_deletion_score == -6.0
        assert aligner.open_right_deletion_score == -7.0
        assert aligner.extend_right_deletion_score == -6.0

    def test_sets_substitution_matrix_and_disables_wildcard(self, substitution_matrix):
        aligner = setup_aligner(substitution_matrix, mode=AlignmentMode.GLOBAL)
        assert aligner.wildcard is None
        # round-trips through Biopython's own equality for Array objects
        assert list(aligner.substitution_matrix.alphabet) == list(substitution_matrix.alphabet)

    def test_default_gap_scores_are_negative(self, substitution_matrix):
        """Sanity check on the documented defaults -- gaps should be
        penalized, not free, out of the box."""
        aligner = setup_aligner(substitution_matrix)
        assert aligner.open_internal_insertion_score < 0
        assert aligner.open_left_insertion_score < 0
