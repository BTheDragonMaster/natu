"""Tests for NATU's two multiple sequence alignment strategies:

- Progressive (UPGMA guide-tree) alignment: the pure profile-alignment
  helpers in natu.progressive_msa, and their integration into
  natu.msa._progressive_msa / natu.msa.calculate_msa(..., progressive=True).
- Center-star alignment (the default when progressive=False): the pure
  helpers in natu.msa (_merge_center_alignment, _star_msa), and their
  integration into natu.msa.calculate_msa(..., progressive=False).

Layout: one test class per function under test, named Test<FunctionName>,
with each case as a method on that class. Uses the shared converter/
make_aligner fixtures from tests/conftest.py (alphabet "p","q","r","s";
MATCH_SCORE=4.0, MISMATCH_SCORE=-1.0) rather than NATU's packaged chemistry
matrices, same as tests/test_pairwise.py.

Every alignment result asserted here was confirmed by actually running the
real code first (DP-based profile alignment, and center-star's guide/merge
logic, are not something to hand-derive), not hand-computed -- see
natu-project-reference's testing conventions.

Regression note: writing TestStarMsa's auto-pick-center case surfaced a real
bug in _star_msa (center_star=None always resolved to index 0, regardless of
actual similarity, because the sum used to pick a center was computed over a
matrix whose diagonal had just been set to -inf -- every row/column sum
therefore included its own -inf and came out -inf, so argmax always landed
on the first index). Confirmed with Barbara and fixed in natu.msa._star_msa
before writing these tests.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from natu.constants import AlignmentMode
from natu.msa import _merge_center_alignment, _progressive_msa, _star_msa, calculate_msa
from natu.progressive_msa import (
    _align_along_tree,
    _build_guide_tree,
    _column_score,
    _profile_profile_align,
)


class TestBuildGuideTree:
    def test_clusters_the_most_similar_pairs_first(self):
        """0/1 are the most similar pair (0.9), 2/3 the next most similar
        (0.8) -- UPGMA must group each pair into its own subtree before
        joining the two subtrees at the root."""
        norm_sims = np.array([
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.8],
            [0.1, 0.1, 0.8, 1.0],
        ], dtype=np.float32)

        tree = _build_guide_tree(norm_sims)

        assert not tree.root.is_leaf()

        def leaf_indices(node) -> set[int]:
            if node.is_leaf():
                return {node.index}
            found: set[int] = set()
            for child in node.children:
                found |= leaf_indices(child)
            return found

        left, right = tree.root.children
        groups = {frozenset(leaf_indices(left)), frozenset(leaf_indices(right))}
        assert groups == {frozenset({0, 1}), frozenset({2, 3})}


class TestColumnScore:
    def test_mean_score_across_a_mixed_column(self, substitution_matrix):
        """col_a = [p, q] (indices 0, 1), col_b = [p, p] (indices 0, 0):
        pairwise scores are p/p=4, p/p=4, q/p=-1, q/p=-1 -> mean 1.5."""
        col_a = np.array([0, 1], dtype=np.int32)
        col_b = np.array([0, 0], dtype=np.int32)

        score = _column_score(col_a, col_b, np.asarray(substitution_matrix), np.int32(-1))

        assert score == pytest.approx(1.5)

    def test_returns_zero_when_one_side_is_entirely_gaps(self, substitution_matrix):
        col_a = np.array([0, 1], dtype=np.int32)
        col_all_gap = np.array([-1, -1], dtype=np.int32)

        score = _column_score(col_a, col_all_gap, np.asarray(substitution_matrix), np.int32(-1))

        assert score == 0.0


class TestProfileProfileAlign:
    def test_identical_single_sequence_profiles_align_with_no_gaps(self, make_aligner, converter):
        aligner = make_aligner()
        prof_a = converter.to_int_array(["p", "q", "r"]).reshape(1, -1)
        prof_b = converter.to_int_array(["p", "q", "r"]).reshape(1, -1)

        merged = _profile_profile_align(prof_a, prof_b, converter.gap_repr, aligner)

        assert merged.shape == (2, 3)
        assert converter.from_int_array(merged[0]) == ["p", "q", "r"]
        assert converter.from_int_array(merged[1]) == ["p", "q", "r"]

    def test_a_missing_middle_monomer_in_one_profile_opens_a_gap_column(self, make_aligner, converter):
        """profile_b is missing the middle monomer of profile_a -- confirmed
        by actually running this that the aligner opens a single gap column
        at that position rather than, say, shifting r out to the end."""
        aligner = make_aligner()
        prof_a = converter.to_int_array(["p", "q", "r"]).reshape(1, -1)
        prof_b = converter.to_int_array(["p", "r"]).reshape(1, -1)

        merged = _profile_profile_align(prof_a, prof_b, converter.gap_repr, aligner)

        assert merged.shape == (2, 3)
        assert converter.from_int_array(merged[0]) == ["p", "q", "r"]
        assert converter.from_int_array(merged[1]) == ["p", None, "r"]


class TestAlignAlongTree:
    def test_merges_leaves_along_a_real_guide_tree(self, make_aligner, converter):
        """3 sequences: seq0=p,q,r and seq1=p,q,s share a p,q prefix and are
        the most similar pair, so they merge first; seq2=s,s,s is the most
        different and merges in last, against the already-merged {1,0}
        profile. Exact row order and resulting column layout confirmed by
        actually running this first, not derived by hand from the DP
        recurrence."""
        aligner = make_aligner()
        int_seqs = [
            converter.to_int_array(["p", "q", "r"]),
            converter.to_int_array(["p", "q", "s"]),
            converter.to_int_array(["s", "s", "s"]),
        ]
        norm_sims = np.array([
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ], dtype=np.float32)
        tree = _build_guide_tree(norm_sims)

        msa, order = _align_along_tree(tree.root, int_seqs, converter.gap_repr, aligner)

        assert order == [2, 1, 0]
        assert msa.shape == (3, 5)

        rows_by_seq_index = {seq_idx: converter.from_int_array(msa[row]) for row, seq_idx in enumerate(order)}
        assert rows_by_seq_index[2] == [None, None, "s", "s", "s"]
        assert rows_by_seq_index[1] == ["p", "q", "s", None, None]
        assert rows_by_seq_index[0] == ["p", "q", "r", None, None]

        # No column can be a gap in every single row -- each merge step
        # always carries at least one side's real symbols into every column
        # it produces.
        assert not np.any(np.all(msa == converter.gap_repr, axis=0))


class TestProgressiveMsaFunction:
    """Tests for natu.msa._progressive_msa directly (the guide-tree-building
    and scoring glue around _align_along_tree), as distinct from the full
    natu.msa.calculate_msa entry point tested below."""

    def test_scores_each_sequence_against_the_first_visited_leaf(self, make_aligner, converter):
        """The returned scores are NOT read off the merged profile -- they
        are a fresh aligner.score() of every sequence against
        row_order[0] (the first-visited leaf), including a self-score for
        that reference sequence itself. Confirmed against a real run."""
        aligner = make_aligner()
        int_seqs = [
            converter.to_int_array(["p", "q", "r"]),
            converter.to_int_array(["p", "q", "s"]),
            converter.to_int_array(["s", "s", "s"]),
        ]
        norm_sims = np.array([
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ], dtype=np.float32)

        msa, scores, order = _progressive_msa(norm_sims, converter, aligner, int_seqs)

        assert order == [2, 1, 0]
        assert scores == pytest.approx([12.0, 3.5, -0.6])


class TestCalculateMsaProgressive:
    """Tests for the public natu.msa.calculate_msa(..., progressive=True)
    entry point: similarity-matrix computation, int-sequence conversion,
    and the progressive path together."""

    def test_produces_an_aligned_msa_with_grounded_scores_and_order(self, make_aligner, converter):
        aligner = make_aligner()
        to_align = [["p", "q", "r"], ["p", "q", "s"], ["s", "s", "s"]]

        aligned, new_order = calculate_msa(aligner, to_align, converter, progressive=True)

        assert new_order == [2, 1, 0]
        assert [seq for _, seq in aligned] == [
            [None, None, "s", "s", "s"],
            ["p", "q", "s", None, None],
            ["p", "q", "r", None, None],
        ]
        assert [score for score, _ in aligned] == pytest.approx([12.0, 3.5, -0.6])

    def test_all_identical_sequences_align_with_no_gaps(self, make_aligner, converter):
        aligner = make_aligner()
        to_align = [["p", "q"], ["p", "q"], ["p", "q"]]

        aligned, new_order = calculate_msa(aligner, to_align, converter, progressive=True)

        assert len(new_order) == 3
        for score, seq in aligned:
            assert seq == ["p", "q"]
            assert score == pytest.approx(8.0)  # self-score: 2 matches * MATCH_SCORE(4)

    def test_single_sequence_is_returned_unchanged_with_its_self_score(self, make_aligner, converter):
        aligner = make_aligner()

        aligned, new_order = calculate_msa(aligner, [["p", "q", "r"]], converter, progressive=True)

        assert new_order == [0]
        assert aligned == [(pytest.approx(12.0), ["p", "q", "r"])]

    def test_on_empty_input(self, make_aligner, converter):
        aligner = make_aligner()

        assert calculate_msa(aligner, [], converter, progressive=True) == ([], [])

    def test_forces_global_mode_and_warns_when_aligner_is_not_global(self, make_aligner, converter, caplog):
        """calculate_msa requires global-mode alignment for any MSA (star or
        progressive) -- a non-global aligner is coerced to 'global' in
        place and a warning is logged, rather than silently producing a
        different kind of alignment."""
        aligner = make_aligner(mode=AlignmentMode.LOCAL)
        to_align = [["p", "q", "r"], ["p", "q", "s"], ["s", "s", "s"]]

        with caplog.at_level(logging.WARNING, logger="natu.msa"):
            aligned, new_order = calculate_msa(aligner, to_align, converter, progressive=True)

        assert aligner.mode == "global"
        assert any("global" in record.message for record in caplog.records)
        # same result as if the aligner had been global from the start
        assert [score for score, _ in aligned] == pytest.approx([12.0, 3.5, -0.6])



class TestMergeCenterAlignment:
    def test_folds_a_leading_insertion_into_a_single_row_msa(self, converter):
        """Grounded from a real _pairwise_alignment(center=p,q,r vs
        query=s,p,q,r): t_a=[None,p,q,r], q_a=[s,p,q,r] -- a leading
        insertion in the query, gap in the center. Folding that into an
        msa that's just the raw (ungapped) center row must open the same
        leading gap in the center's own row."""
        center = converter.to_int_array(["p", "q", "r"])
        msa = center.reshape(1, -1)
        t_a = np.array([converter.gap_repr, 0, 1, 2], dtype=np.int32)  # None, p, q, r
        q_a = np.array([3, 0, 1, 2], dtype=np.int32)  # s, p, q, r

        merged = _merge_center_alignment(msa, t_a, q_a, converter.gap_repr)

        assert merged.shape == (2, 4)
        assert converter.from_int_array(merged[0]) == [None, "p", "q", "r"]
        assert converter.from_int_array(merged[1]) == ["s", "p", "q", "r"]

    def test_raises_when_the_new_center_alignment_extends_beyond_the_existing_msa(self, converter):
        """t_a has 2 real symbols but msa is only 1 column wide -- the
        second symbol has nowhere left to land."""
        msa = converter.to_int_array(["p"]).reshape(1, -1)
        t_a = converter.to_int_array(["p", "q"])
        q_a = converter.to_int_array(["p", "r"])

        with pytest.raises(ValueError, match="extends beyond existing MSA"):
            _merge_center_alignment(msa, t_a, q_a, converter.gap_repr)

    def test_raises_when_the_new_center_alignment_disagrees_with_the_existing_msa(self, converter):
        """t_a's second symbol (r) doesn't match what's actually in the
        existing msa's center row at that column (q) -- an internal
        consistency check, since every call here is meant to realign the
        SAME center sequence the existing msa was built from."""
        msa = converter.to_int_array(["p", "q"]).reshape(1, -1)
        t_a = converter.to_int_array(["p", "r"])
        q_a = converter.to_int_array(["p", "s"])

        with pytest.raises(ValueError, match="center alignment mismatch"):
            _merge_center_alignment(msa, t_a, q_a, converter.gap_repr)


class TestStarMsa:
    def test_aligns_every_sequence_to_an_explicit_center(self, make_aligner, converter):
        """center_star=0: seq1 (index 1, p,q,s) gets a query-side insertion
        (s) relative to the p,q,r center, and seq2 (index 2, s,p,q,r) gets
        a leading insertion -- each merged into the growing msa
        independently, both against the raw ungapped center. Exact scores,
        row order, and column layout confirmed by actually running this."""
        aligner = make_aligner()
        int_seqs = [
            converter.to_int_array(["p", "q", "r"]),
            converter.to_int_array(["p", "q", "s"]),
            converter.to_int_array(["s", "p", "q", "r"]),
        ]
        sims = np.array([
            [12.0, 7.3, 11.8],
            [7.3, 12.0, 7.1],
            [11.8, 7.1, 16.0],
        ], dtype=np.float32)

        msa, scores, indices = _star_msa(sims, center_star=0, converter=converter, aligner=aligner, int_seqs=int_seqs)

        assert indices == [0, 2, 1]
        assert scores == pytest.approx([12.0, 11.8, 7.3])
        assert [converter.from_int_array(row) for row in msa] == [
            [None, "p", "q", None, "r"],
            ["s", "p", "q", None, "r"],
            [None, "p", "q", "s", None],
        ]

    def test_auto_pick_selects_the_sequence_most_similar_to_all_others(self, make_aligner, converter):
        """Regression test for the bug described in this module's docstring:
        with this similarity matrix, index 2 is similar to both index 0
        and index 1 (0.95 each), while 0 and 1 are only similar to 2, not
        to each other (0.1) -- so 2 has the highest total similarity and
        must be the auto-picked center. Before the fix, center_star=None
        always resolved to index 0 regardless of this matrix."""
        aligner = make_aligner()
        int_seqs = [converter.to_int_array(["p", "q"]) for _ in range(3)]
        sims = np.array([
            [1.0, 0.1, 0.95],
            [0.1, 1.0, 0.95],
            [0.95, 0.95, 1.0],
        ], dtype=np.float32)

        msa, scores, indices = _star_msa(
            sims, center_star=None, converter=converter, aligner=aligner, int_seqs=int_seqs
        )

        assert indices[0] == 2
        assert scores == pytest.approx([8.0, 8.0, 8.0])  # identical sequences: all self-score


class TestCalculateMsaStar:
    """Tests for the public natu.msa.calculate_msa(..., progressive=False)
    entry point (the default): similarity-matrix computation, int-sequence
    conversion, and the center-star path together."""

    def test_auto_picked_center_produces_grounded_scores_and_order(self, make_aligner, converter):
        aligner = make_aligner()
        to_align = [["p", "q", "r"], ["p", "q", "s"], ["s", "p", "q", "r"]]

        aligned, new_order = calculate_msa(aligner, to_align, converter)

        assert new_order == [0, 2, 1]
        assert [seq for _, seq in aligned] == [
            [None, "p", "q", None, "r"],
            ["s", "p", "q", None, "r"],
            [None, "p", "q", "s", None],
        ]
        assert [score for score, _ in aligned] == pytest.approx([12.0, 11.8, 7.3])

    def test_an_explicit_center_star_overrides_the_auto_pick(self, make_aligner, converter):
        aligner = make_aligner()
        to_align = [["p", "q", "r"], ["p", "q", "s"], ["s", "p", "q", "r"]]

        aligned, new_order = calculate_msa(aligner, to_align, converter, center_star=1)

        assert new_order == [1, 0, 2]
        assert [seq for _, seq in aligned] == [
            [None, "p", "q", None, None, "s"],
            [None, "p", "q", None, "r", None],
            ["s", "p", "q", "r", None, None],
        ]
        assert [score for score, _ in aligned] == pytest.approx([12.0, 7.3, 7.1])

    def test_all_identical_sequences_align_with_no_gaps(self, make_aligner, converter):
        aligner = make_aligner()
        to_align = [["p", "q"], ["p", "q"], ["p", "q"]]

        aligned, new_order = calculate_msa(aligner, to_align, converter)

        assert len(new_order) == 3
        for score, seq in aligned:
            assert seq == ["p", "q"]
            assert score == pytest.approx(8.0)

    def test_single_sequence_is_returned_unchanged_with_its_self_score(self, make_aligner, converter):
        aligner = make_aligner()

        aligned, new_order = calculate_msa(aligner, [["p", "q", "r"]], converter)

        assert new_order == [0]
        assert aligned == [(pytest.approx(12.0), ["p", "q", "r"])]

    def test_on_empty_input(self, make_aligner, converter):
        aligner = make_aligner()

        assert calculate_msa(aligner, [], converter) == ([], [])

    def test_forces_global_mode_and_warns_when_aligner_is_not_global(self, make_aligner, converter, caplog):
        aligner = make_aligner(mode=AlignmentMode.LOCAL)
        to_align = [["p", "q", "r"], ["p", "q", "s"], ["s", "p", "q", "r"]]

        with caplog.at_level(logging.WARNING, logger="natu.msa"):
            aligned, new_order = calculate_msa(aligner, to_align, converter)

        assert aligner.mode == "global"
        assert any("global" in record.message for record in caplog.records)
        assert [score for score, _ in aligned] == pytest.approx([12.0, 11.8, 7.3])
