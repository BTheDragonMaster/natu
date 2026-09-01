"""Tests for natu.matrix: the pure pandas/numpy helpers (get_average_self_score,
get_average_non_self_score, get_mean_scores, add_wildcard,
rescale_similarity_matrix), the chemistry-heavy expand_substitution_matrix,
check_structures, and build_substitution_matrix -- the last two now covered
using real project data (src/natu/data/structures/{smiles,paras_smiles}.tsv
and the real packaged substitution matrix files), the same grounding
approach used throughout test_variants.py: every expected value below was
confirmed by actually running the real function first, not computed by hand.

Layout: one test class per function under test, named Test<FunctionName>,
with each case as a method on that class.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from natu.constants import AlignmentConfiguration, SubstitutionMatrix
from natu.matrix import (
    add_wildcard,
    build_substitution_matrix,
    check_structures,
    expand_substitution_matrix,
    get_average_non_self_score,
    get_average_non_self_score_from_name,
    get_average_self_score,
    get_mean_scores,
    rescale_similarity_matrix,
)
from pikachu.general import read_smiles

# Real substrate SMILES, copied verbatim from
# src/natu/data/structures/smiles.tsv (same source used in test_variants.py).
# ALANINE and GLYCINE also happen to exist under the exact same name in the
# packaged PARAS reference set (src/natu/data/structures/paras_smiles.tsv),
# which TestBuildSubstitutionMatrix's PARAS_BASED tests rely on.
ALANINE = "C[C@@H](C(=O)O)N"
GLYCINE = "NCC(=O)O"
SERINE = "C([C@@H](C(=O)O)N)O"
ASPARTIC_ACID = "C([C@@H](C(=O)O)N)C(=O)O"


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


class TestExpandSubstitutionMatrix:
    """expand_substitution_matrix takes an existing base_alphabet x
    base_alphabet substitution matrix and, for each substrate already in it,
    adds one column/row per tailoring variant (D-/NMe-/NMe-D- forms) computed
    for that substrate. Each new pair's score is base_score + chirality_bonus
    + methylation_bonus, where base_score is looked up from the *existing*
    matrix between the two variants' underlying base substrates, and the
    bonuses come from the tailoring config's scoring tables, keyed by each
    variant's own Chirality/NMethylation.

    Every expected cell value below was confirmed by actually running this
    function against the real ECFP tailoring config and real alanine/glycine
    SMILES first, not computed by hand and hoped for.
    """

    @pytest.fixture
    def tailoring_config(self) -> dict:
        config = yaml.safe_load(AlignmentConfiguration.ECFP.read_text())
        return config["tailoring"]

    @pytest.fixture
    def base_matrix(self) -> pd.DataFrame:
        # A tiny, made-up but self-consistent base matrix -- only its shape
        # and labels need to be real (alanine/glycine), the actual numbers
        # just need to be distinct enough to catch a mixed-up lookup.
        return pd.DataFrame(
            [[10.0, 2.0], [2.0, 8.0]],
            index=["alanine", "glycine"],
            columns=["alanine", "glycine"],
        )

    @pytest.fixture
    def smiles_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "mini_smiles.tsv"
        path.write_text(f"substrate\tsmiles\nalanine\t{ALANINE}\nglycine\t{GLYCINE}\n")
        return path

    def test_returns_the_same_object_when_tailoring_config_is_incomplete(self, base_matrix):
        # missing both "chirality" and "n_methylation" sections entirely
        result = expand_substitution_matrix(base_matrix, {}, smiles_file=None)
        assert result is base_matrix

    def test_returns_the_same_object_when_no_smiles_file_is_given(self, base_matrix, tailoring_config):
        result = expand_substitution_matrix(base_matrix, tailoring_config, smiles_file=None)
        assert result is base_matrix

    def test_logs_when_tailoring_is_enabled_but_no_smiles_file_is_given(self, base_matrix, tailoring_config, caplog):
        """There is no exception raised anywhere for "tailoring-aware
        scoring is on but no --smiles was given" -- expansion is just
        silently skipped (see the test above). This is the only signal
        that a matrix expected to have tailoring bonuses baked into it
        was actually returned as-is: a debug-level log line, previously
        untested. If the caller's alphabet doesn't happen to need any of
        the never-generated variants, nothing downstream ever notices;
        if it does, the failure surfaces much later and generically, as
        natu.cli's "symbol ... was not found in the substitution matrix
        alphabet" (see test_align.py's
        test_smiles_flag_expands_the_alphabet_with_tailoring_variants)."""
        assert tailoring_config["chirality"]["chirality_aware_scoring"] is True
        assert tailoring_config["n_methylation"]["n_methylation_aware_scoring"] is True

        with caplog.at_level(logging.DEBUG, logger="natu.matrix"):
            expand_substitution_matrix(base_matrix, tailoring_config, smiles_file=None)

        assert "No SMILES file provided" in caplog.text
        assert "cannot expand substitution matrix" in caplog.text

    def test_does_not_log_the_no_smiles_message_when_tailoring_is_off(self, caplog):
        """The 'no SMILES file' debug message is conditioned on tailoring
        actually being enabled -- with both scoring flags off, smiles_file
        being None is a complete non-issue and must not log that message
        (a different, earlier debug line covers a genuinely incomplete
        config; this is a fully *complete* config with scoring disabled)."""
        base_matrix = pd.DataFrame(
            [[10.0, 2.0], [2.0, 8.0]], index=["alanine", "glycine"], columns=["alanine", "glycine"]
        )
        tailoring_config = {
            "chirality": {"chirality_aware_scoring": False},
            "n_methylation": {"n_methylation_aware_scoring": False},
        }

        with caplog.at_level(logging.DEBUG, logger="natu.matrix"):
            result = expand_substitution_matrix(base_matrix, tailoring_config, smiles_file=None)

        assert result is base_matrix
        assert "No SMILES file provided" not in caplog.text

    def test_does_not_mutate_the_input_matrix(self, base_matrix, tailoring_config, smiles_file):
        original = base_matrix.copy()
        expand_substitution_matrix(base_matrix, tailoring_config, smiles_file)
        pd.testing.assert_frame_equal(base_matrix, original)

    def test_adds_one_column_and_row_per_generated_variant(self, base_matrix, tailoring_config, smiles_file):
        """alanine (chiral, N-H) gets NMe-/D-/NMe-D- variants; glycine
        (achiral) only gets an NMe- variant -- same as compute_variants'
        own behavior, since this just calls get_structure_variants under
        the hood."""
        expanded = expand_substitution_matrix(base_matrix, tailoring_config, smiles_file)

        expected_columns = ["alanine", "glycine", "NMe-alanine", "D-alanine", "NMe-D-alanine", "NMe-glycine"]
        assert list(expanded.columns) == expected_columns
        assert list(expanded.index) == expected_columns  # rows and columns in the same order
        assert expanded.shape == (6, 6)

    def test_new_pair_score_is_base_score_plus_chirality_and_methylation_bonus(
        self, base_matrix, tailoring_config, smiles_file
    ):
        """D-alanine vs alanine: base_score is alanine's own self-score (10.0,
        since D-alanine's base is alanine and alanine's own variant object is
        itself the base substrate); chirality bonus is l_d = -5.0 (D-alanine
        is Chirality.D, and the *base* 'alanine' variant object ends up
        Chirality.L once compute_variants has run, not X); methylation bonus
        is n_n = 1.0 (both are NMethylation.N). Total: 10.0 - 5.0 + 1.0 = 6.0.
        """
        expanded = expand_substitution_matrix(base_matrix, tailoring_config, smiles_file)

        assert expanded.loc["alanine", "D-alanine"] == pytest.approx(6.0)
        assert expanded.loc["D-alanine", "alanine"] == pytest.approx(6.0)  # symmetric

    def test_new_pair_score_uses_the_correct_base_pair_for_cross_substrate_terms(
        self, base_matrix, tailoring_config, smiles_file
    ):
        """D-alanine vs glycine: base_score comes from the *original* matrix's
        alanine/glycine entry (2.0), chirality bonus is d_x = 0.0 (glycine's
        own chirality is X), methylation bonus is n_n = 1.0 (glycine's base
        variant is also NMethylation.N). Total: 2.0 + 0.0 + 1.0 = 3.0.
        """
        expanded = expand_substitution_matrix(base_matrix, tailoring_config, smiles_file)

        assert expanded.loc["glycine", "D-alanine"] == pytest.approx(3.0)
        assert expanded.loc["D-alanine", "glycine"] == pytest.approx(3.0)

    def test_variant_self_score_stacks_both_bonuses(self, base_matrix, tailoring_config, smiles_file):
        """D-alanine vs itself: base_score = alanine self-score (10.0),
        chirality bonus d_d = 5.0, methylation bonus n_n = 1.0 -> 16.0."""
        expanded = expand_substitution_matrix(base_matrix, tailoring_config, smiles_file)
        assert expanded.loc["D-alanine", "D-alanine"] == pytest.approx(16.0)

    def test_two_variants_of_different_substrates_combine_both_bonus_tables(
        self, base_matrix, tailoring_config, smiles_file
    ):
        """D-alanine (Chirality.D, NMethylation.N) vs NMe-glycine
        (Chirality.X, NMethylation.Y): base_score is alanine/glycine's
        original entry (2.0), chirality bonus d_x = 0.0, methylation bonus
        n_y = -1.0 -> 1.0."""
        expanded = expand_substitution_matrix(base_matrix, tailoring_config, smiles_file)
        assert expanded.loc["D-alanine", "NMe-glycine"] == pytest.approx(1.0)
        assert expanded.loc["NMe-glycine", "D-alanine"] == pytest.approx(1.0)


class TestCheckStructures:
    """check_structures raises on duplicate substrate *names* and
    debug-logs (does not raise) when two differently-named substrates turn
    out to be structurally identical.

    Note: the duplicate-name branch (`if name_1 == name_2: raise
    ValueError(...)`) is unreachable through this function's own signature
    -- name_to_structure is a dict, and a dict cannot contain two entries
    with the same key in the first place, so `names` (its .keys()) can never
    contain a repeat for the loop to find. Not tested here, since there is
    no legitimate way to construct the input that would trigger it.
    """

    def test_returns_none_for_structurally_distinct_monomers(self):
        result = check_structures({"alanine": read_smiles(ALANINE), "glycine": read_smiles(GLYCINE)})
        assert result is None

    def test_logs_a_debug_warning_for_structurally_identical_monomers_with_different_names(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="natu.matrix"):
            check_structures({
                "aspartic acid": read_smiles(ASPARTIC_ACID),
                "aspartic acid branched": read_smiles(ASPARTIC_ACID),
            })

        assert "aspartic acid" in caplog.text
        assert "aspartic acid branched" in caplog.text
        assert "structurally identical" in caplog.text


class TestBuildSubstitutionMatrix:
    """build_substitution_matrix reads a SMILES file, computes a matrix over
    its substrates according to matrix_type, and writes it to out_file as a
    tab-separated table -- it has no return value, so every test here reads
    the written file back to check the result.
    """

    @staticmethod
    def _write_smiles_file(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
        path = tmp_path / "smiles.tsv"
        lines = ["substrate\tsmiles"] + [f"{name}\t{smi}" for name, smi in rows]
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_match_mismatch_writes_an_identity_matrix(self, tmp_path: Path):
        smiles_file = self._write_smiles_file(tmp_path, [("alanine", ALANINE), ("glycine", GLYCINE)])
        out_file = tmp_path / "out.txt"

        build_substitution_matrix(smiles_file, SubstitutionMatrix.MATCH_MISMATCH, out_file)

        result = pd.read_csv(out_file, sep="\t", index_col=0)
        assert list(result.index) == ["alanine", "glycine"]
        assert result.loc["alanine", "alanine"] == pytest.approx(1.0)
        assert result.loc["glycine", "glycine"] == pytest.approx(1.0)
        assert result.loc["alanine", "glycine"] == pytest.approx(0.0)
        assert result.loc["glycine", "alanine"] == pytest.approx(0.0)

    def test_ecfp_writes_a_rescaled_pikachu_jaccard_similarity_matrix(self, tmp_path: Path):
        """Values confirmed by actually running build_substitution_matrix
        against these exact three substrates first. Deterministic: PIKAChU's
        Jaccard fingerprint similarity has no random component, and
        rescale_similarity_matrix (tested on its own above) always maps the
        single highest similarity in the matrix -- here, every self-pair,
        since nothing can be more similar to a structure than itself -- to
        target_max=7.0, and the single lowest pairwise similarity to
        target_min=-5.0."""
        smiles_file = self._write_smiles_file(
            tmp_path, [("alanine", ALANINE), ("glycine", GLYCINE), ("serine", SERINE)]
        )
        out_file = tmp_path / "out.txt"

        build_substitution_matrix(smiles_file, SubstitutionMatrix.ECFP, out_file)

        result = pd.read_csv(out_file, sep="\t", index_col=0)
        for name in ("alanine", "glycine", "serine"):
            assert result.loc[name, name] == pytest.approx(7.0)
        # alanine/glycine is the least similar pair of the three -> hits target_min
        assert result.loc["alanine", "glycine"] == pytest.approx(-5.0)
        assert result.loc["glycine", "alanine"] == pytest.approx(-5.0)
        assert result.loc["alanine", "serine"] == pytest.approx(-2.142857, abs=1e-5)
        assert result.loc["glycine", "serine"] == pytest.approx(-4.571429, abs=1e-5)

    def test_featmorgan_writes_a_rescaled_rdkit_tanimoto_similarity_matrix(self, tmp_path: Path):
        """Same rescaling logic as ECFP, but similarity comes from rdkit
        Morgan/feature fingerprints (Tanimoto) instead of PIKAChU/Jaccard --
        deterministic for the same reason, and the min/max pairing is
        different (glycine/serine is least similar here, not alanine/glycine)."""
        smiles_file = self._write_smiles_file(
            tmp_path, [("alanine", ALANINE), ("glycine", GLYCINE), ("serine", SERINE)]
        )
        out_file = tmp_path / "out.txt"

        build_substitution_matrix(smiles_file, SubstitutionMatrix.FEATMORGAN, out_file)

        result = pd.read_csv(out_file, sep="\t", index_col=0)
        for name in ("alanine", "glycine", "serine"):
            assert result.loc[name, name] == pytest.approx(7.0)
        assert result.loc["glycine", "serine"] == pytest.approx(-5.0)
        assert result.loc["serine", "glycine"] == pytest.approx(-5.0)
        assert result.loc["alanine", "glycine"] == pytest.approx(-1.571429, abs=1e-5)
        assert result.loc["alanine", "serine"] == pytest.approx(-1.470588, abs=1e-5)

    def test_raises_for_an_unrecognized_matrix_type(self, tmp_path: Path):
        """SubstitutionMatrix only has 4 members (MATCH_MISMATCH, PARAS_BASED,
        ECFP, FEATMORGAN) and the if/elif chain in build_substitution_matrix
        handles all 4, so its final `else: raise ValueError(...)` can't
        actually be reached by any real SubstitutionMatrix value -- Python
        doesn't enforce the matrix_type: SubstitutionMatrix type hint at
        runtime, though, so passing something else still exercises the
        guard itself."""
        smiles_file = self._write_smiles_file(tmp_path, [("alanine", ALANINE)])

        with pytest.raises(ValueError, match="Unknown matrix type"):
            build_substitution_matrix(smiles_file, "not_a_real_matrix_type", tmp_path / "out.txt")


@pytest.fixture(scope="module")
def reference_matrix() -> pd.DataFrame:
    """The real packaged PARAS-based substitution matrix, loaded once and
    reused read-only across TestBuildSubstitutionMatrixParasBased -- a
    plain module-level fixture rather than a class-scoped one, since
    pytest's support for a classmethod-based class-scoped fixture is
    version-dependent (it silently fails to register as a fixture at all
    on some pytest versions instead of just warning)."""
    with SubstitutionMatrix.PARAS_BASED.open() as handle:
        return pd.read_csv(handle, sep="\t", index_col=0)


class TestBuildSubstitutionMatrixParasBased:
    """PARAS_BASED mode scores substrate pairs by matching each substrate to
    the packaged PARAS reference set (structurally, via is_equivalent, not
    just by name) and looking up that reference pair's real score in the
    packaged paras_based.txt matrix -- falling back to averages from that
    same reference matrix for any substrate PARAS doesn't recognize.

    Expected values are computed from the real reference matrix inside each
    test (via the same get_average_self_score/get_average_non_self_score/
    get_average_non_self_score_from_name helpers build_substitution_matrix
    itself uses) rather than hardcoded, so these stay correct if the
    packaged paras_based.txt is ever regenerated.
    """

    @staticmethod
    def _write_smiles_file(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
        path = tmp_path / "smiles.tsv"
        lines = ["substrate\tsmiles"] + [f"{name}\t{smi}" for name, smi in rows]
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_exact_name_and_structure_match_uses_the_real_reference_score_directly(
        self, tmp_path: Path, reference_matrix: pd.DataFrame
    ):
        # "alanine" exists under the exact same name and SMILES in the
        # packaged PARAS reference set, so it matches directly.
        smiles_file = self._write_smiles_file(tmp_path, [("alanine", ALANINE), ("glycine", GLYCINE)])
        out_file = tmp_path / "out.txt"

        build_substitution_matrix(smiles_file, SubstitutionMatrix.PARAS_BASED, out_file)
        result = pd.read_csv(out_file, sep="\t", index_col=0)

        assert result.loc["alanine", "alanine"] == pytest.approx(reference_matrix.loc["alanine", "alanine"])
        assert result.loc["alanine", "glycine"] == pytest.approx(reference_matrix.loc["alanine", "glycine"])
        assert result.loc["glycine", "alanine"] == pytest.approx(reference_matrix.loc["glycine", "alanine"])

    def test_a_differently_named_but_structurally_identical_substrate_matches_by_structure(
        self, tmp_path: Path, reference_matrix: pd.DataFrame
    ):
        # Same SMILES as alanine, but a name that doesn't exist in the
        # reference set at all -- is_equivalent still finds the match.
        smiles_file = self._write_smiles_file(tmp_path, [("L-alanine-synonym", ALANINE)])
        out_file = tmp_path / "out.txt"

        build_substitution_matrix(smiles_file, SubstitutionMatrix.PARAS_BASED, out_file)
        result = pd.read_csv(out_file, sep="\t", index_col=0)

        assert result.loc["L-alanine-synonym", "L-alanine-synonym"] == pytest.approx(
            reference_matrix.loc["alanine", "alanine"]
        )

    def test_falls_back_to_the_matched_sides_reference_row_mean_when_the_other_side_is_unmatched(
        self, tmp_path: Path, reference_matrix: pd.DataFrame
    ):
        # a plain long alkane: not present in, and structurally unrelated
        # to, anything in the packaged PARAS reference set
        novel_compound = "CCCCCCCCCCCCCCCCCC"
        smiles_file = self._write_smiles_file(
            tmp_path, [("alanine", ALANINE), ("totally novel test compound", novel_compound)]
        )
        out_file = tmp_path / "out.txt"

        build_substitution_matrix(smiles_file, SubstitutionMatrix.PARAS_BASED, out_file)
        result = pd.read_csv(out_file, sep="\t", index_col=0)

        expected = get_average_non_self_score_from_name(reference_matrix, "alanine")
        assert result.loc["alanine", "totally novel test compound"] == pytest.approx(expected)
        assert result.loc["totally novel test compound", "alanine"] == pytest.approx(expected)

    def test_falls_back_to_the_global_reference_non_self_mean_when_neither_side_matches(
        self, tmp_path: Path, reference_matrix: pd.DataFrame
    ):
        novel_compound = "CCCCCCCCCCCCCCCCCC"
        smiles_file = self._write_smiles_file(tmp_path, [("totally novel test compound", novel_compound)])
        out_file = tmp_path / "out.txt"

        build_substitution_matrix(smiles_file, SubstitutionMatrix.PARAS_BASED, out_file)
        result = pd.read_csv(out_file, sep="\t", index_col=0)

        expected_self = get_average_self_score(reference_matrix)
        expected_non_self = get_average_non_self_score(reference_matrix)
        # a lone, unmatched substrate: its "self" entry also falls back
        # (nothing to look up directly), using the *self*-score average,
        # not the non-self one.
        assert result.loc["totally novel test compound", "totally novel test compound"] == pytest.approx(
            expected_self
        )
        assert expected_self != pytest.approx(expected_non_self)  # sanity: these two are genuinely different numbers
