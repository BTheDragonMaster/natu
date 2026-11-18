"""Tests for the closest-structure fallback of PARAS-based substitution matrices.

Substrates without an equivalent in the PARAS dataset can borrow the scores of their structurally closest PARAS
substrate (config section 'unknown_substrates'), instead of getting average scores.
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
    get_unknown_substrate_settings,
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
def test_unknown_substrate_borrows_scores_of_closest(tmp_path, method):
    settings = {"use_closest_structure": True, "similarity": method, "min_similarity": 0.0,
                "base_penalty": 0.2, "penalty_scale": 5.0}
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
