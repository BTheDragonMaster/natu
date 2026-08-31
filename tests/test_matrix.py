"""Tests for the pure pandas/numpy helpers in natu.matrix:
get_average_self_score, get_average_non_self_score, get_mean_scores,
add_wildcard, and rescale_similarity_matrix.

Out of scope here: build_substitution_matrix, expand_substitution_matrix,
and check_structures, which need real SMILES/structure data (pikachu/rdkit
chemistry objects) to exercise meaningfully -- those are integration-level
and are better covered with real project data than synthetic fixtures.
Flagged as not covered by this starter suite rather than silently skipped.

Layout: one test class per function under test, named Test<FunctionName>,
with each case as a method on that class.
"""

from __future__ import annotations

import pandas as pd
import pytest

from natu.matrix import (
    add_wildcard,
    get_average_non_self_score,
    get_average_non_self_score_from_name,
    get_average_self_score,
    get_mean_scores,
    rescale_similarity_matrix,
)


@pytest.fixture
def small_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [[4.0, -1.0, -1.0], [-1.0, 4.0, -1.0], [-1.0, -1.0, 4.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )


class TestGetAverageSelfScore:
    def test_is_the_diagonal_mean(self, small_matrix):
        assert get_average_self_score(small_matrix) == pytest.approx(4.0)


class TestGetAverageNonSelfScore:
    def test_excludes_the_diagonal(self, small_matrix):
        assert get_average_non_self_score(small_matrix) == pytest.approx(-1.0)


class TestGetMeanScores:
    def test_excludes_each_row_own_diagonal_entry(self, small_matrix):
        means = get_mean_scores(small_matrix)
        assert means.to_dict() == pytest.approx({"a": -1.0, "b": -1.0, "c": -1.0})


class TestGetAverageNonSelfScoreFromName:
    def test_returns_the_row_mean_excluding_that_name(self, small_matrix):
        assert get_average_non_self_score_from_name(small_matrix, "a") == pytest.approx(-1.0)


class TestAddWildcard:
    def test_is_a_noop_when_disabled(self, small_matrix):
        config = {"wildcard": {"wildcard_for_unknowns": False, "wildcard_character": "X"}}
        result = add_wildcard(small_matrix, config)
        assert result is small_matrix  # returned completely unchanged, not even copied

    def test_adds_a_square_row_and_column(self, small_matrix):
        config = {"wildcard": {"wildcard_for_unknowns": True, "wildcard_character": "X"}}
        result = add_wildcard(small_matrix, config)

        assert result.shape == (4, 4)
        assert "X" in result.columns and "X" in result.index
        # wildcard's self-score is the overall average self-score
        assert result.loc["X", "X"] == pytest.approx(4.0)
        # wildcard's score against a real substrate is that substrate's own
        # non-self mean (symmetric both ways)
        assert result.loc["a", "X"] == pytest.approx(-1.0)
        assert result.loc["X", "a"] == pytest.approx(-1.0)
        assert result.isna().sum().sum() == 0

    def test_rejects_a_character_that_already_exists(self):
        df = pd.DataFrame([[4.0, -1.0], [-1.0, 4.0]], index=["a", "X"], columns=["a", "X"])
        config = {"wildcard": {"wildcard_for_unknowns": True, "wildcard_character": "X"}}

        with pytest.raises(ValueError, match="already exists in the matrix"):
            add_wildcard(df, config)

    def test_does_not_mutate_the_input_dataframe(self, small_matrix):
        original = small_matrix.copy()
        config = {"wildcard": {"wildcard_for_unknowns": True, "wildcard_character": "X"}}
        add_wildcard(small_matrix, config)
        pd.testing.assert_frame_equal(small_matrix, original)


class TestRescaleSimilarityMatrix:
    def test_maps_min_and_max_exactly(self):
        sim = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["a", "b"], columns=["a", "b"])
        rescaled = rescale_similarity_matrix(sim, target_min=-5, target_max=7)

        assert rescaled.loc["a", "a"] == pytest.approx(7.0)
        assert rescaled.loc["a", "b"] == pytest.approx(-5.0)
        assert list(rescaled.index) == ["a", "b"]

    def test_is_monotonic(self):
        """A purely affine transform must preserve rank order."""
        sim = pd.DataFrame(
            [[1.0, 0.5, 0.1], [0.5, 1.0, 0.3], [0.1, 0.3, 1.0]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        rescaled = rescale_similarity_matrix(sim, target_min=-2, target_max=13)

        assert rescaled.loc["a", "b"] > rescaled.loc["a", "c"]
        assert rescaled.loc["b", "c"] > rescaled.loc["a", "c"]

    def test_rejects_mismatched_labels(self):
        sim = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["a", "b"], columns=["b", "a"])
        with pytest.raises(AssertionError):
            rescale_similarity_matrix(sim)

    def test_rejects_all_identical_similarities(self):
        sim = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], index=["a", "b"], columns=["a", "b"])
        with pytest.raises(AssertionError, match="cannot rescale"):
            rescale_similarity_matrix(sim)
