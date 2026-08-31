"""Tests for natu.variants: the chemistry-heavy module that computes
D-/N-methylated structure variants for each substrate ahead of matrix
building.

Unlike the rest of the suite, these tests exercise real PIKAChU chemistry
(via pikachu.general.read_smiles) instead of a synthetic p/q/r/s alphabet --
the functions here genuinely need real amino-acid backbones (a chiral
alpha-carbon, a backbone N-H, etc.) to exercise anything meaningful. All
SMILES strings below are taken verbatim from the project's own packaged
substrate list (src/natu/data/structures/smiles.tsv), not invented, and
every expected value (D-enantiomer SMILES, N-methylated SMILES, variant
names) was confirmed by actually running the real functions against real
PIKAChU/real project chemistry before being written into an assertion here
-- none of this is guessed.

Layout: one test class per function under test, named Test<FunctionName>,
with each case as a method on that class. StructureCollection (a class, not
a function) gets its own TestStructureCollection, the same convention used
for Converter in test_pairwise.py.

Out of scope: nothing chemistry-related was skipped here, but
StructureCollection/compute_variants tests intentionally build Variant
objects with structure=None wherever the code path under test never reads
.structure (e.g. get_variant_name's n_methylation-only branch, or
StructureCollection.write_variants/write_smiles), to keep those cases
decoupled from chemistry setup.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pikachu.general import read_smiles, structure_to_smiles

from natu.constants import AlignmentConfiguration, SubstitutionMatrix
from natu.variants import (
    Chirality,
    Modification,
    NMethylation,
    StructureCollection,
    Variant,
    compute_variants,
    epimerization,
    get_d_structure,
    get_nme_structure,
    get_structure_variants,
    get_variant_name,
    has_chiral_beta_carbon,
    is_equivalent,
    methylation,
    modify,
    parse_smiles,
    remove_equivalents,
    substrates_from_matrix,
)

# --- Real substrate SMILES, copied verbatim from
# src/natu/data/structures/smiles.tsv -------------------------------------

ALANINE = "C[C@@H](C(=O)O)N"  # L-alanine: chiral alpha-carbon, no beta-carbon
GLYCINE = "NCC(=O)O"  # achiral: no chiral alpha-carbon at all
THREONINE = "C[C@H]([C@@H](C(=O)O)N)O"  # chiral alpha- AND beta-carbon
PROLINE = "C1C[C@H](NC1)C(=O)O"  # secondary ring amine (still has one N-H)
ASPARTIC_ACID = "C([C@@H](C(=O)O)N)C(=O)O"
ASPARTIC_ACID_BRANCHED = "C([C@@H](C(=O)O)N)C(=O)O"  # same structure, excluded name
NON_AMINO_ACID = "C1=CC(=C(C(=C1)O)O)C(=O)O"  # 2,3-dihydroxybenzoic acid: no N-C-C(=O)O backbone at all


@pytest.fixture
def tailoring_config() -> dict:
    """Real packaged tailoring config (chirality + n_methylation sections)."""
    config = yaml.safe_load(AlignmentConfiguration.ECFP.read_text())
    return config["tailoring"]


class TestEpimerization:
    def test_flips_clockwise_to_counterclockwise_and_back(self):
        structure = read_smiles(ALANINE)
        alpha_carbon = next(a for a in structure.atoms.values() if a.chiral)
        original = alpha_carbon.chiral
        assert original in ("clockwise", "counterclockwise")

        epimerization(alpha_carbon)
        assert alpha_carbon.chiral != original

        epimerization(alpha_carbon)
        assert alpha_carbon.chiral == original

    def test_leaves_a_non_chiral_atom_as_none(self):
        structure = read_smiles(ALANINE)
        achiral_atom = next(
            a for a in structure.atoms.values() if a.type == "C" and not a.chiral
        )
        epimerization(achiral_atom)
        assert achiral_atom.chiral is None


class TestMethylation:
    def test_methylates_the_backbone_nitrogen(self):
        """Direct test of the low-level function get_nme_structure wraps;
        expected SMILES confirmed via get_nme_structure(alanine) beforehand."""
        structure = read_smiles(ALANINE)
        from natu.variants import N_AMINO_ACID
        from pikachu.reactions.functional_groups import find_atoms

        nitrogen = find_atoms(N_AMINO_ACID, structure)[0]
        methylated = methylation(nitrogen, structure)

        assert structure_to_smiles(methylated) == "CN[C@H](C(O)=O)C"

    def test_raises_when_target_atom_has_no_hydrogen_to_replace(self):
        structure = read_smiles(ALANINE)
        # the carboxylic-acid carbon: bonded to =O, -O(H), -C, no H of its own
        carboxyl_carbon = next(
            a for a in structure.atoms.values()
            if a.type == "C" and any(b.type == "double" for b in a.bonds)
        )
        assert carboxyl_carbon.get_neighbour("H") is None

        with pytest.raises(Exception, match="Can't methylate this atom"):
            methylation(carboxyl_carbon, structure)


class TestParseSmiles:
    def test_parses_name_smiles_pairs_skipping_the_header(self, tmp_path: Path):
        smiles_file = tmp_path / "mini_smiles.tsv"
        smiles_file.write_text(f"substrate\tsmiles\nalanine\t{ALANINE}\nglycine\t{GLYCINE}\n")

        result = parse_smiles(smiles_file)

        assert result == {"alanine": ALANINE, "glycine": GLYCINE}

    def test_skips_blank_lines(self, tmp_path: Path):
        smiles_file = tmp_path / "mini_smiles.tsv"
        smiles_file.write_text(f"substrate\tsmiles\nalanine\t{ALANINE}\n\nglycine\t{GLYCINE}\n")

        result = parse_smiles(smiles_file)

        assert result == {"alanine": ALANINE, "glycine": GLYCINE}

    def test_header_only_file_returns_empty_dict(self, tmp_path: Path):
        smiles_file = tmp_path / "mini_smiles.tsv"
        smiles_file.write_text("substrate\tsmiles\n")

        assert parse_smiles(smiles_file) == {}


class TestGetDStructure:
    def test_returns_the_d_enantiomer_of_an_l_amino_acid(self):
        d_structure = get_d_structure(read_smiles(ALANINE))
        assert d_structure is not None
        assert structure_to_smiles(d_structure) == "C[C@@H](N)C(O)=O"

    def test_returns_none_for_an_achiral_amino_acid(self):
        assert get_d_structure(read_smiles(GLYCINE)) is None

    def test_returns_none_when_there_is_no_amino_acid_backbone(self):
        assert get_d_structure(read_smiles(NON_AMINO_ACID)) is None


class TestGetNmeStructure:
    def test_methylates_the_backbone_nitrogen_of_an_amino_acid(self):
        nme_structure = get_nme_structure(read_smiles(ALANINE))
        assert nme_structure is not None
        assert structure_to_smiles(nme_structure) == "CN[C@H](C(O)=O)C"

    def test_methylates_a_secondary_ring_amine(self):
        """Proline's backbone N is secondary (in-ring) but still carries one
        H, so it's a real, chemically valid methylation target."""
        nme_structure = get_nme_structure(read_smiles(PROLINE))
        assert nme_structure is not None
        assert structure_to_smiles(nme_structure) == "CN1[C@H](C(O)=O)CCC1"

    def test_returns_none_when_there_is_no_amino_acid_nitrogen(self):
        assert get_nme_structure(read_smiles(NON_AMINO_ACID)) is None


class TestIsEquivalent:
    def test_true_for_two_independent_parses_of_the_same_smiles(self):
        assert is_equivalent(read_smiles(ALANINE), read_smiles(ALANINE)) is True

    def test_false_for_unrelated_structures(self):
        assert is_equivalent(read_smiles(ALANINE), read_smiles(GLYCINE)) is False

    def test_false_for_enantiomers(self):
        """Enantiomers are different structures for this purpose -- chirality
        is part of what find_substructures compares, so D-alanine is NOT
        considered equivalent to L-alanine."""
        d_alanine = get_d_structure(read_smiles(ALANINE))
        assert is_equivalent(read_smiles(ALANINE), d_alanine) is False


class TestSubstratesFromMatrix:
    def test_reads_the_header_row_as_the_substrate_list(self, tmp_path: Path):
        matrix_file = tmp_path / "ecfp.txt"
        matrix_file.write_text(SubstitutionMatrix.ECFP.read_text())

        substrates = substrates_from_matrix(str(matrix_file))

        assert isinstance(substrates, list)
        assert "alanine" in substrates
        assert len(substrates) > 1


class TestModify:
    def test_true_when_no_exclusion_substring_matches(self):
        assert modify("aspartic acid", ["branched", "2-aminoadipic acid"]) is True

    def test_false_when_name_contains_an_excluded_substring(self):
        assert modify("aspartic acid branched", ["branched", "2-aminoadipic acid"]) is False

    def test_true_when_exclusions_list_is_empty(self):
        assert modify("anything", []) is True


class TestHasChiralBetaCarbon:
    def test_true_for_threonine(self):
        """Threonine's beta-carbon bears both a methyl and a hydroxyl --
        a real, chiral second stereocentre (the L-allo-/D-allo- distinction
        this drives is a real, documented NRPS nomenclature case)."""
        assert has_chiral_beta_carbon(read_smiles(THREONINE)) is True

    def test_false_for_alanine(self):
        """Alanine's beta-carbon is just a methyl group -- not a stereocentre."""
        assert has_chiral_beta_carbon(read_smiles(ALANINE)) is False

    def test_false_for_an_achiral_amino_acid(self):
        assert has_chiral_beta_carbon(read_smiles(GLYCINE)) is False


class TestGetVariantName:
    def test_adds_the_d_prefix_when_there_is_no_existing_chirality_prefix(self, tailoring_config):
        d_structure = get_d_structure(read_smiles(ALANINE))
        name = get_variant_name("alanine", d_structure, tailoring_config, [Modification.chirality])
        assert name == "D-alanine"

    def test_adds_the_allo_prefix_when_the_beta_carbon_is_also_chiral(self, tailoring_config):
        d_structure = get_d_structure(read_smiles(THREONINE))
        name = get_variant_name("threonine", d_structure, tailoring_config, [Modification.chirality])
        assert name == "D-allo-threonine"

    def test_adds_the_nme_prefix_for_n_methylation(self, tailoring_config):
        # the n_methylation-only branch never reads `structure`, so a dummy
        # value keeps this case decoupled from chemistry setup
        name = get_variant_name("alanine", None, tailoring_config, [Modification.n_methylation])
        assert name == "NMe-alanine"


class TestComputeVariants:
    def test_alanine_produces_base_d_nme_and_nme_d_variants(self, tailoring_config):
        collection = compute_variants("alanine", ALANINE, tailoring_config)

        assert collection.base_structure.name == "alanine"
        assert collection.base_structure.chirality == Chirality.L
        assert collection.base_structure.n_methylation == NMethylation.N

        by_name = {v.name: v for v in collection.variants}
        assert set(by_name) == {"NMe-alanine", "D-alanine", "NMe-D-alanine"}
        assert by_name["D-alanine"].smiles == "C[C@@H](N)C(O)=O"
        assert by_name["D-alanine"].chirality == Chirality.D
        assert by_name["NMe-alanine"].n_methylation == NMethylation.Y
        assert all(v.base is collection.base_structure for v in collection.variants)

    def test_respects_the_chirality_exclusion_list(self, tailoring_config):
        """'aspartic acid branched' matches the config's chirality exclude
        list ("branched"), so it should get an N-methylated variant but no
        D-/NMe-D- variant."""
        collection = compute_variants(
            "aspartic acid branched", ASPARTIC_ACID_BRANCHED, tailoring_config
        )

        by_name = {v.name: v.smiles for v in collection.variants}
        assert by_name == {"NMe-aspartic acid branched": "CN[C@H](C(O)=O)CC(O)=O"}

    def test_achiral_substrate_still_gets_n_methylated(self, tailoring_config):
        collection = compute_variants("glycine", GLYCINE, tailoring_config)

        by_name = {v.name: v.smiles for v in collection.variants}
        assert by_name == {"NMe-glycine": "CNCC(O)=O"}
        assert collection.base_structure.chirality == Chirality.X  # never set for an achiral base


class TestRemoveEquivalents:
    def test_raises_on_duplicate_substrate_names(self, tailoring_config):
        collection_1 = compute_variants("aspartic acid", ASPARTIC_ACID, tailoring_config)
        collection_2 = compute_variants("aspartic acid", ASPARTIC_ACID, tailoring_config)

        with pytest.raises(ValueError, match="Duplicate substrates"):
            remove_equivalents([collection_1, collection_2])

    def test_keeps_differently_named_structurally_identical_entries(self, tailoring_config):
        """Confirmed intended behavior (see remove_equivalents' docstring):
        two base structures that are structurally identical but carry
        different names -- like 'aspartic acid' and 'aspartic acid
        branched', which share a SMILES in the packaged data -- are only
        debug-logged, not deduplicated. Only a *variant* whose own name
        collides with another collection's base name (i.e. a substrate that
        was redundantly listed under its own auto-generated variant name)
        gets removed.
        """
        collection_1 = compute_variants("aspartic acid", ASPARTIC_ACID, tailoring_config)
        collection_2 = compute_variants(
            "aspartic acid branched", ASPARTIC_ACID_BRANCHED, tailoring_config
        )

        filtered = remove_equivalents([collection_1, collection_2])

        assert [c.base_structure.name for c in filtered] == [
            "aspartic acid",
            "aspartic acid branched",
        ]


class TestGetStructureVariants:
    @pytest.fixture
    def smiles_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "mini_smiles.tsv"
        path.write_text(f"substrate\tsmiles\nalanine\t{ALANINE}\nglycine\t{GLYCINE}\n")
        return path

    def test_computes_variants_for_every_substrate_in_the_file(self, smiles_file, tailoring_config):
        variants = get_structure_variants(smiles_file, tailoring_config)
        by_name = {v.name: v.smiles for v in variants}
        assert by_name == {
            "alanine": ALANINE,
            "NMe-alanine": "CN[C@H](C(O)=O)C",
            "D-alanine": "C[C@@H](N)C(O)=O",
            "NMe-D-alanine": "CN[C@@H](C(O)=O)C",
            "glycine": GLYCINE,
            "NMe-glycine": "CNCC(O)=O",
        }

    def test_filters_to_the_requested_substrates(self, smiles_file, tailoring_config):
        variants = get_structure_variants(smiles_file, tailoring_config, substrates=["alanine"])
        by_name = {v.name: v.smiles for v in variants}
        assert by_name == {
            "alanine": ALANINE,
            "NMe-alanine": "CN[C@H](C(O)=O)C",
            "D-alanine": "C[C@@H](N)C(O)=O",
            "NMe-D-alanine": "CN[C@@H](C(O)=O)C",
        }

    def test_raises_keyerror_for_an_unknown_requested_substrate(self, smiles_file, tailoring_config):
        with pytest.raises(KeyError):
            get_structure_variants(smiles_file, tailoring_config, substrates=["not_a_real_substrate"])


class TestStructureCollection:
    def _variant(self, name: str, modifications: list[Modification]) -> Variant:
        return Variant(
            name=name,
            smiles=f"smiles-for-{name}",
            structure=None,
            base=None,
            chirality=Chirality.X,
            n_methylation=NMethylation.X,
            modifications=modifications,
        )

    def test_get_variant_from_modifications_matches_by_modification_set(self):
        base = self._variant("base", [])
        d_variant = self._variant("D-base", [Modification.chirality])
        nme_variant = self._variant("NMe-base", [Modification.n_methylation])
        collection = StructureCollection(base, [d_variant, nme_variant])

        assert collection.get_variant_from_modifications([Modification.chirality]) is d_variant
        assert collection.get_variant_from_modifications([Modification.n_methylation]) is nme_variant

    def test_get_variant_from_modifications_returns_none_when_nothing_matches(self):
        base = self._variant("base", [])
        collection = StructureCollection(base, [])
        assert collection.get_variant_from_modifications([Modification.chirality]) is None

    def test_write_variants_uses_tab_placeholders_for_missing_variants(self, tmp_path: Path, tailoring_config):
        """Grounded against the real compute_variants output for 'aspartic
        acid branched' (no D-/NMe-D- variant, since it's chirality-excluded)."""
        collection = compute_variants(
            "aspartic acid branched", ASPARTIC_ACID_BRANCHED, tailoring_config
        )
        out_file = tmp_path / "variants.tsv"

        collection.write_variants(str(out_file))

        assert out_file.read_text() == (
            "aspartic acid branched\tC([C@@H](C(=O)O)N)C(=O)O\t"
            "\t\t"
            "NMe-aspartic acid branched\tCN[C@H](C(O)=O)CC(O)=O"
            "\t\t\n"
        )

    def test_write_variants_full_row_when_all_three_variants_exist(self, tmp_path: Path, tailoring_config):
        collection = compute_variants("alanine", ALANINE, tailoring_config)
        out_file = tmp_path / "variants.tsv"

        collection.write_variants(str(out_file))

        assert out_file.read_text() == (
            "alanine\tC[C@@H](C(=O)O)N\t"
            "D-alanine\tC[C@@H](N)C(O)=O\t"
            "NMe-alanine\tCN[C@H](C(O)=O)C\t"
            "NMe-D-alanine\tCN[C@@H](C(O)=O)C\n"
        )

    def test_write_smiles_writes_base_then_every_variant(self, tmp_path: Path, tailoring_config):
        collection = compute_variants("alanine", ALANINE, tailoring_config)
        out_file = tmp_path / "smiles.tsv"

        collection.write_smiles(str(out_file))

        lines = out_file.read_text().splitlines()
        assert lines[0] == "alanine\tC[C@@H](C(=O)O)N"
        assert set(lines[1:]) == {
            "NMe-alanine\tCN[C@H](C(O)=O)C",
            "D-alanine\tC[C@@H](N)C(O)=O",
            "NMe-D-alanine\tCN[C@@H](C(O)=O)C",
        }
