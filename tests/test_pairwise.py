"""Tests for natu.pairwise: Converter, glocal end-gap configuration, and the
low-level _pairwise_alignment entry point.

Layout: one test class per function (or per class, for Converter) under
test, named Test<FunctionName>, with each case as a method on that class.
"""

from __future__ import annotations

import numpy as np
import pytest

from natu.constants import AlignmentMode
from natu.pairwise import (
    Converter,
    _pairwise_alignment,
    configure_glocal_end_gaps,
    replace_unknowns_with_wildcards,
    strip_glocal_free_overhang,
)


class TestConverter:
    def test_round_trips_a_sequence(self, converter):
        seq = ["p", "q", "r", "s"]
        int_array = converter.to_int_array(seq)
        assert int_array.tolist() == [0, 1, 2, 3]
        assert converter.from_int_array(int_array) == seq

    def test_from_int_array_renders_gap_repr_as_none(self, converter):
        # gap_repr defaults to -1 and must round-trip to None, not a real symbol
        int_array = np.array([0, converter.gap_repr, 1], dtype=np.int32)
        assert converter.from_int_array(int_array) == ["p", None, "q"]

    def test_to_int_array_of_empty_sequence(self, converter):
        result = converter.to_int_array([])
        assert result.tolist() == []


class TestReplaceUnknownsWithWildcards:
    def test_replaces_only_unknown_items(self):
        alphabet = ["ala", "gly"]
        sequence = ["ala", "unknown_thing", "gly"]
        result = replace_unknowns_with_wildcards(alphabet, sequence, "X")
        assert result == ["ala", "X", "gly"]

    def test_on_empty_sequence(self):
        assert replace_unknowns_with_wildcards(["ala"], [], "X") == []

    def test_all_unknown(self):
        result = replace_unknowns_with_wildcards([], ["a", "b"], "X")
        assert result == ["X", "X"]


class TestConfigureGlocalEndGaps:
    """
    Biopython naming note (see the function's own docstring): NATU always
    passes t as seqA and q as seqB, so "t longer" frees the *deletion*
    end-gap attributes and "q longer" frees the *insertion* ones.
    """

    def test_frees_deletion_side_when_t_longer(self, make_aligner):
        aligner = make_aligner(mode=AlignmentMode.GLOCAL, open_end_gap_score=-3.0, extend_end_gap_score=-2.0)
        configure_glocal_end_gaps(aligner, len_t=5, len_q=2, open_end_gap_score=-3.0, extend_end_gap_score=-2.0)

        assert aligner.open_left_deletion_score == 0.0
        assert aligner.extend_left_deletion_score == 0.0
        assert aligner.open_right_deletion_score == 0.0
        assert aligner.extend_right_deletion_score == 0.0
        # the non-free side must stay at the configured (penalized) score
        assert aligner.open_left_insertion_score == -3.0
        assert aligner.extend_left_insertion_score == -2.0
        assert aligner.open_right_insertion_score == -3.0
        assert aligner.extend_right_insertion_score == -2.0

    def test_frees_insertion_side_when_q_longer(self, make_aligner):
        aligner = make_aligner(mode=AlignmentMode.GLOCAL, open_end_gap_score=-3.0, extend_end_gap_score=-2.0)
        configure_glocal_end_gaps(aligner, len_t=2, len_q=5, open_end_gap_score=-3.0, extend_end_gap_score=-2.0)

        assert aligner.open_left_insertion_score == 0.0
        assert aligner.extend_left_insertion_score == 0.0
        assert aligner.open_right_insertion_score == 0.0
        assert aligner.extend_right_insertion_score == 0.0
        assert aligner.open_left_deletion_score == -3.0
        assert aligner.extend_left_deletion_score == -2.0
        assert aligner.open_right_deletion_score == -3.0
        assert aligner.extend_right_deletion_score == -2.0

    def test_frees_nothing_when_lengths_equal(self, make_aligner):
        aligner = make_aligner(mode=AlignmentMode.GLOCAL, open_end_gap_score=-3.0, extend_end_gap_score=-2.0)
        configure_glocal_end_gaps(aligner, len_t=3, len_q=3, open_end_gap_score=-3.0, extend_end_gap_score=-2.0)

        for attr in (
            "open_left_insertion_score", "extend_left_insertion_score",
            "open_right_insertion_score", "extend_right_insertion_score",
            "open_left_deletion_score", "extend_left_deletion_score",
            "open_right_deletion_score", "extend_right_deletion_score",
        ):
            expected = -3.0 if attr.startswith("open") else -2.0
            assert getattr(aligner, attr) == expected, attr


class TestStripGlocalFreeOverhang:
    def test_returns_unchanged_when_lengths_equal(self):
        t_a = np.array([0, 1, 2], dtype=np.int32)
        q_a = np.array([0, 1, 2], dtype=np.int32)
        out_t, out_q = strip_glocal_free_overhang(t_a, q_a, np.int32(-1), len_t=3, len_q=3)
        assert out_t is t_a
        assert out_q is q_a

    def test_strips_leading_and_trailing_free_gaps(self):
        # t is the SHORTER sequence (len_t=3 < len_q=6): its free leading/trailing
        # overhang shows up as gap_repr runs at the very edges of t_a.
        gap = np.int32(-1)
        t_a = np.array([gap, gap, 0, 1, 2, gap], dtype=np.int32)
        q_a = np.array([5, 6, 0, 1, 2, 7], dtype=np.int32)

        out_t, out_q = strip_glocal_free_overhang(t_a, q_a, gap, len_t=3, len_q=6)

        assert out_t.tolist() == [0, 1, 2]
        assert out_q.tolist() == [0, 1, 2]

    def test_uses_q_as_shorter_side_when_q_shorter(self):
        gap = np.int32(-1)
        t_a = np.array([5, 6, 0, 1, 2, 7], dtype=np.int32)
        q_a = np.array([gap, gap, 0, 1, 2, gap], dtype=np.int32)

        out_t, out_q = strip_glocal_free_overhang(t_a, q_a, gap, len_t=6, len_q=3)

        assert out_t.tolist() == [0, 1, 2]
        assert out_q.tolist() == [0, 1, 2]

    def test_leaves_a_genuine_internal_gap_alone(self):
        # An internal gap (surrounded by real residues) must survive -- only
        # edge-run gaps get stripped.
        gap = np.int32(-1)
        t_a = np.array([0, gap, 1], dtype=np.int32)  # shorter side, no edge gaps at all
        q_a = np.array([0, 2, 1], dtype=np.int32)

        out_t, out_q = strip_glocal_free_overhang(t_a, q_a, gap, len_t=2, len_q=3)

        assert out_t.tolist() == [0, gap, 1]
        assert out_q.tolist() == [0, 2, 1]


class TestPairwiseAlignment:
    def test_global_identical_sequences(self, make_aligner, converter):
        aligner = make_aligner(mode=AlignmentMode.GLOBAL)
        t = converter.to_int_array(["p", "q", "r"])
        q = converter.to_int_array(["p", "q", "r"])

        score, t_a, q_a = _pairwise_alignment(aligner, t, q, gap_repr=converter.gap_repr, trim=True)

        assert score == pytest.approx(12.0)  # 3 matches * MATCH_SCORE(4)
        assert converter.from_int_array(t_a) == ["p", "q", "r"]
        assert converter.from_int_array(q_a) == ["p", "q", "r"]

    def test_local_trim_false_pads_to_full_length(self, make_aligner, converter):
        aligner = make_aligner(mode=AlignmentMode.LOCAL)
        t = converter.to_int_array(["p", "p", "q", "q", "p"])
        q = converter.to_int_array(["q", "q"])

        score, t_a, q_a = _pairwise_alignment(aligner, t, q, gap_repr=converter.gap_repr, trim=False)

        assert score == pytest.approx(8.0)  # 2 matches * MATCH_SCORE(4)
        assert len(t_a) == len(t)  # padded out to t's full original length
        assert converter.from_int_array(t_a) == ["p", "p", "q", "q", "p"]
        assert converter.from_int_array(q_a) == [None, None, "q", "q", None]

    def test_local_trim_true_keeps_only_aligned_core(self, make_aligner, converter):
        aligner = make_aligner(mode=AlignmentMode.LOCAL)
        t = converter.to_int_array(["p", "p", "q", "q", "p"])
        q = converter.to_int_array(["q", "q"])

        score, t_a, q_a = _pairwise_alignment(aligner, t, q, gap_repr=converter.gap_repr, trim=True)

        assert len(t_a) == 2  # only the matched "q","q" core, no padding
        assert converter.from_int_array(t_a) == ["q", "q"]
        assert converter.from_int_array(q_a) == ["q", "q"]

    def test_returns_none_when_best_local_score_is_zero(self, make_aligner, converter):
        """
        Documented Biopython gotcha: aligner.align() reports ZERO alignments
        (not "one alignment scoring 0") whenever the best possible local
        alignment scores exactly 0 -- entirely plausible whenever two
        sequences share no matching symbol at all under a substitution
        matrix with negative mismatch scores, since the empty alignment
        (score 0) then beats every non-empty one.
        """
        aligner = make_aligner(mode=AlignmentMode.LOCAL)
        t = converter.to_int_array(["p", "p", "p"])
        q = converter.to_int_array(["q", "q", "q"])  # shares no symbol with t

        assert len(aligner.align(seqA=t, seqB=q)) == 0  # confirms the Biopython behavior itself
        result = _pairwise_alignment(aligner, t, q, gap_repr=converter.gap_repr, trim=True)

        assert result is None


# Note: natu.pairwise used to also expose an align() wrapper around
# _pairwise_alignment that unpacked its result without a None guard (a real
# crash risk for the same "best local score is exactly 0" case tested
# above, since it -- unlike search.py and network.cluster_sequences -- never
# checked for None). Barbara confirmed align() wasn't called anywhere in the
# codebase and removed it, so the tests that covered it have been removed
# too rather than kept around for dead code.
