"""Tests for the closest-structure fallback of PARAS-based substitution matrices.

Substrates without an equivalent in the PARAS dataset can use the scores of their structurally closest PARAS
substrate (config section 'unknown_substrates'), instead of getting average scores. In the default 'interpolate'
mode, the scores move from NATU's average scores (similarity at the threshold) to the closest substrate's scores
(identical structure); in 'subtract' mode, the closest substrate's scores are copied.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pikachu.general import read_smiles

from natu.constants import SubstitutionMatrix
from natu.matrix import (
    UNKNOWN_SUBSTRATE_DEFAULTS,
    build_substitution_matrix,
    closest_structure_penalty,
    find_closest_substrates,
    get_average_non_self_score_from_name,
    get_average_self_score,
    get_similarity_threshold,
    get_unknown_substrate_settings,
    interpolation_weight,
    make_fingerprints,
    strip_stereochemistry,
)

LEUCINE = "CC(C)C[C@@H](C(=O)O)N"
GLYCINE = "NCC(=O)O"
PHENYLALANINE = "C1=CC=C(C=C1)C[C@@H](C(=O)O)N"
HOMOLEUCINE = "CC(C)CC[C@@H](C(=O)O)N"          # one CH2 longer than leucine
FLUOROLEUCINE = "FCC(C)C[C@@H](C(=O)O)N"         # not in the PARAS dataset
D_LEUCINE = "CC(C)C[C@H](C(=O)O)N"

REFERENCE = {"leucine": LEUCINE, "glycine": GLYCINE, "phenylalanine": PHENYLALANINE}


def _structures(name_to_smiles):
    return {n: read_smiles(s) for n, s in name_to_smiles.items()}


class TestClosestStructurePenalty:
    def test_equivalent_pair_only_pays_the_base_penalty(self):
        assert closest_structure_penalty(1.0, 1.0, 0.2, 5.0) == pytest.approx(0.2)

    def test_scales_with_dissimilarity(self):
        assert closest_structure_penalty(0.8, 1.0, 0.2, 5.0) == pytest.approx(1.2)

    def test_adds_dissimilarities_of_both_substrates(self):
        assert closest_structure_penalty(0.8, 0.6, 0.2, 5.0) == pytest.approx(0.2 + 5.0 * 0.6)

    def test_fixed_penalty_when_scale_is_zero(self):
        assert closest_structure_penalty(0.4, 1.0, 0.2, 0.0) == pytest.approx(0.2)


class TestInterpolationWeight:
    def test_zero_at_and_below_threshold(self):
        assert interpolation_weight(0.4, 0.4) == 0.0
        assert interpolation_weight(0.2, 0.4) == 0.0

    def test_one_for_identical_structure(self):
        assert interpolation_weight(1.0, 0.4) == pytest.approx(1.0)

    def test_linear_in_between(self):
        assert interpolation_weight(0.7, 0.4) == pytest.approx(0.5)

    def test_threshold_one_only_accepts_identical(self):
        assert interpolation_weight(1.0, 1.0) == 1.0
        assert interpolation_weight(0.99, 1.0) == 0.0


class TestUnknownSubstrateSettings:
    def test_defaults_without_config(self):
        assert get_unknown_substrate_settings(None) == UNKNOWN_SUBSTRATE_DEFAULTS

    def test_config_overrides_only_given_keys(self):
        settings = get_unknown_substrate_settings({"unknown_substrates": {"similarity": "featmorgan"}})
        assert settings["similarity"] == "featmorgan"
        assert settings["min_similarity"] == UNKNOWN_SUBSTRATE_DEFAULTS["min_similarity"]

    def test_rejects_unknown_similarity_method(self):
        with pytest.raises(ValueError):
            get_unknown_substrate_settings({"unknown_substrates": {"similarity": "maccs"}})

    def test_rejects_threshold_outside_zero_one(self):
        with pytest.raises(ValueError):
            get_unknown_substrate_settings({"unknown_substrates": {"min_similarity": 1.5}})

    def test_accepts_auto_threshold_and_numbers(self):
        assert get_unknown_substrate_settings({"unknown_substrates": {"min_similarity": "auto"}})["min_similarity"] == "auto"
        assert get_unknown_substrate_settings({"unknown_substrates": {"min_similarity": 0.4}})["min_similarity"] == 0.4

    def test_rejects_other_threshold_strings(self):
        with pytest.raises(ValueError):
            get_unknown_substrate_settings({"unknown_substrates": {"min_similarity": "high"}})

    def test_rejects_unknown_penalty_mode(self):
        with pytest.raises(ValueError):
            get_unknown_substrate_settings({"unknown_substrates": {"penalty_mode": "multiply"}})

    def test_rejects_percentile_outside_zero_hundred(self):
        with pytest.raises(ValueError):
            get_unknown_substrate_settings({"unknown_substrates": {"auto_percentile": 120}})

    def test_defaults_are_interpolate_and_auto(self):
        settings = get_unknown_substrate_settings(None)
        assert settings["penalty_mode"] == "interpolate"
        assert settings["min_similarity"] == "auto"

    def test_packaged_paras_config_keeps_average_scores_by_default(self):
        import yaml
        config = yaml.safe_load(SubstitutionMatrix.PARAS_BASED.get_config().read_text())
        settings = get_unknown_substrate_settings(config)
        assert settings["use_closest_structure"] is False
        assert set(settings) == set(UNKNOWN_SUBSTRATE_DEFAULTS)


@pytest.mark.parametrize("method", ["ecfp", "featmorgan"])
class TestFindClosestSubstrates:
    def test_finds_the_most_similar_reference(self, method):
        query = {"homoleucine": HOMOLEUCINE}
        closest = find_closest_substrates(["homoleucine"], query, _structures(query),
                                          REFERENCE, _structures(REFERENCE), method=method)
        name, similarity = closest["homoleucine"]
        assert name == "leucine"
        assert 0.0 < similarity < 1.0

    def test_identical_structure_has_similarity_one(self, method):
        query = {"my leucine": LEUCINE}
        closest = find_closest_substrates(["my leucine"], query, _structures(query),
                                          REFERENCE, _structures(REFERENCE), method=method)
        assert closest["my leucine"] == ("leucine", pytest.approx(1.0))

    def test_ties_are_broken_alphabetically(self, method):
        reference = {"b-leucine": LEUCINE, "a-leucine": LEUCINE}
        query = {"q": LEUCINE}
        closest = find_closest_substrates(["q"], query, _structures(query),
                                          reference, _structures(reference), method=method)
        assert closest["q"][0] == "a-leucine"


    def test_ignore_chirality_maps_d_form_onto_l_form(self, method):
        query = {"D-leucine": D_LEUCINE}
        closest = find_closest_substrates(["D-leucine"], query, _structures(query),
                                          REFERENCE, _structures(REFERENCE), method=method,
                                          ignore_chirality=True)
        assert closest["D-leucine"] == ("leucine", pytest.approx(1.0))


def test_strip_stereochemistry():
    assert strip_stereochemistry("C/C=C/C[C@@H](C)[C@H](O)N") == "CC=CC[CH](C)[CH](O)N"


@pytest.mark.parametrize("method", ["ecfp", "featmorgan"])
def test_similarity_threshold_is_a_percentile_of_reference_pairs(method):
    reference = {"leucine": LEUCINE, "glycine": GLYCINE, "phenylalanine": PHENYLALANINE,
                 "homoleucine": HOMOLEUCINE}
    fingerprints = make_fingerprints(reference, _structures(reference), method)
    lowest = get_similarity_threshold(fingerprints, method, percentile=0)
    middle = get_similarity_threshold(fingerprints, method, percentile=50)
    highest = get_similarity_threshold(fingerprints, method, percentile=100)
    assert 0.0 <= lowest <= middle <= highest <= 1.0
    # the most similar pair of these four is leucine / homoleucine
    closest = find_closest_substrates(["homoleucine"], {"homoleucine": HOMOLEUCINE},
                                      _structures({"homoleucine": HOMOLEUCINE}),
                                      {k: v for k, v in reference.items() if k != "homoleucine"},
                                      _structures({k: v for k, v in reference.items() if k != "homoleucine"}),
                                      method=method)
    assert highest == pytest.approx(closest["homoleucine"][1])


def test_similarity_threshold_needs_two_references():
    fingerprints = make_fingerprints({"leucine": LEUCINE}, _structures({"leucine": LEUCINE}), "ecfp")
    with pytest.raises(ValueError):
        get_similarity_threshold(fingerprints, "ecfp")


def test_find_closest_rejects_unknown_method():
    with pytest.raises(ValueError):
        find_closest_substrates([], {}, {}, REFERENCE, _structures(REFERENCE), method="maccs")


# --- build_substitution_matrix, PARAS_BASED -----------------------------------------------------------------------

def _paras_df():
    with SubstitutionMatrix.PARAS_BASED.open() as handle:
        return pd.read_csv(handle, sep="\t", index_col=0)


def _build(tmp_path: Path, unknown_settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    smiles = tmp_path / "smiles.tsv"
    smiles.write_text("name\tsmiles\n"
                      f"my-leucine\t{LEUCINE}\n"
                      f"my-glycine\t{GLYCINE}\n"
                      f"fluoroleucine\t{FLUOROLEUCINE}\n")
    out = tmp_path / "matrix.tsv"
    build_substitution_matrix(smiles, SubstitutionMatrix.PARAS_BASED, out,
                              {"unknown_substrates": unknown_settings})
    matrix = pd.read_csv(out, sep="\t", index_col=0)
    mapping = pd.read_csv(tmp_path / "matrix_paras_mapping.tsv", sep="\t", index_col=0, keep_default_na=False)
    return matrix, mapping


@pytest.mark.parametrize("method", ["ecfp", "featmorgan"])
def test_subtract_mode_copies_scores_of_closest(tmp_path, method):
    settings = {"use_closest_structure": True, "similarity": method, "min_similarity": 0.0,
                "penalty_mode": "subtract", "base_penalty": 0.2, "penalty_scale": 5.0}
    matrix, mapping = _build(tmp_path, settings)
    paras = _paras_df()

    assert mapping.loc["my-leucine", "paras_substrate"] == "leucine"
    assert mapping.loc["my-leucine", "status"] == "equivalent"
    assert mapping.loc["fluoroleucine", "status"] == "closest"

    closest = mapping.loc["fluoroleucine", "paras_substrate"]
    similarity = float(mapping.loc["fluoroleucine", "similarity"])

    # self score and scores against other substrates are copied from the closest PARAS substrate
    assert matrix.loc["fluoroleucine", "fluoroleucine"] == pytest.approx(paras.loc[closest, closest])
    if closest != "glycine":
        assert matrix.loc["fluoroleucine", "my-glycine"] == pytest.approx(paras.loc[closest, "glycine"])

    # against the substrate it borrows from: self score minus the similarity-scaled penalty
    expected = paras.loc[closest, "leucine"]
    if closest == "leucine":
        expected -= closest_structure_penalty(similarity, 1.0, 0.2, 5.0)
    # (the mapping file rounds the similarity to 4 decimals)
    assert matrix.loc["fluoroleucine", "my-leucine"] == pytest.approx(expected, abs=1e-3)
    assert matrix.loc["my-leucine", "fluoroleucine"] == pytest.approx(expected, abs=1e-3)

    # substrates with an equivalent are unchanged
    assert matrix.loc["my-leucine", "my-glycine"] == pytest.approx(paras.loc["leucine", "glycine"])


@pytest.mark.parametrize("method", ["ecfp", "featmorgan"])
def test_interpolate_mode_moves_from_average_to_closest_scores(tmp_path, method):
    threshold = 0.2
    settings = {"use_closest_structure": True, "similarity": method, "min_similarity": threshold,
                "penalty_mode": "interpolate", "base_penalty": 0.2}
    matrix, mapping = _build(tmp_path, settings)
    paras = _paras_df()

    assert mapping.loc["fluoroleucine", "status"] == "closest"
    closest = mapping.loc["fluoroleucine", "paras_substrate"]
    weight = float(mapping.loc["fluoroleucine", "weight"])
    similarity = float(mapping.loc["fluoroleucine", "similarity"])
    assert weight == pytest.approx(interpolation_weight(similarity, threshold), abs=1e-3)
    assert 0.0 < weight < 1.0

    def between(average, borrowed):
        return average + (borrowed - average) * weight

    # self score: between NATU's average self score and the closest substrate's self score
    assert matrix.loc["fluoroleucine", "fluoroleucine"] == pytest.approx(
        between(get_average_self_score(paras), paras.loc[closest, closest]), abs=1e-3)

    # against a substrate with an equivalent: between that substrate's average score and the borrowed score
    target = paras.loc[closest, "leucine"] - (0.2 if closest == "leucine" else 0.0)
    expected = between(get_average_non_self_score_from_name(paras, "leucine"), target)
    assert matrix.loc["fluoroleucine", "my-leucine"] == pytest.approx(expected, abs=1e-3)
    assert matrix.loc["my-leucine", "fluoroleucine"] == pytest.approx(expected, abs=1e-3)

    # substrates with an equivalent are unchanged
    assert matrix.loc["my-leucine", "my-glycine"] == pytest.approx(paras.loc["leucine", "glycine"])


def test_threshold_falls_back_to_average_scores(tmp_path):
    settings = {"use_closest_structure": True, "similarity": "ecfp", "min_similarity": 1.0}
    matrix, mapping = _build(tmp_path, settings)
    paras = _paras_df()

    assert mapping.loc["fluoroleucine", "status"] == "average"
    assert matrix.loc["fluoroleucine", "fluoroleucine"] == pytest.approx(get_average_self_score(paras))
    assert matrix.loc["fluoroleucine", "my-leucine"] == pytest.approx(
        get_average_non_self_score_from_name(paras, "leucine"))


def test_disabled_fallback_keeps_average_scores(tmp_path):
    matrix, mapping = _build(tmp_path, {"use_closest_structure": False})
    paras = _paras_df()

    assert mapping.loc["fluoroleucine", "status"] == "average"
    assert matrix.loc["fluoroleucine", "fluoroleucine"] == pytest.approx(get_average_self_score(paras))
